import streamlit as st
import os
from pathlib import Path
import json
import datetime
import subprocess

from dcm_pipeline import run_dcm_pipeline

# -----------------
# Global Constants & Paths
# -----------------
OUT_DIR = Path("Output")
DB_DIR = OUT_DIR / "Patients"
DB_DIR.mkdir(parents=True, exist_ok=True)

EXP_DB = OUT_DIR / "Experimenters.json"
if not EXP_DB.exists():
    with open(EXP_DB, "w") as f:
        json.dump(["Dr. Smith", "Unknown"], f)

ASSESSOR_DB = OUT_DIR / "Assessors.json"
if not ASSESSOR_DB.exists():
    with open(ASSESSOR_DB, "w") as f:
        json.dump(["Clinical Assessor A", "Clinical Assessor B"], f)

FRAME_QUALITY_PRESETS = {
    "Distant subject": {
        "edge_margin": 0.005,
        "min_leg_vis": 0.05,
        "adaptive_leg_vis": True,
    },
    "Standard": {
        "edge_margin": 0.02,
        "min_leg_vis": 0.30,
        "adaptive_leg_vis": False,
    },
}

# -----------------
# Helper Functions
# -----------------
def load_json_list(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json_list(path, data):
    with open(path, "w") as f:
        json.dump(data, f)

def get_patients():
    patients = []
    for p in DB_DIR.iterdir():
        if p.is_dir() and (p / "profile.json").exists():
            patients.append(p.name)
    return sorted(patients)

def load_patient(pid):
    with open(DB_DIR / pid / "profile.json", "r") as f:
        return json.load(f)
        
def save_patient(pid, data):
    pdir = DB_DIR / pid
    pdir.mkdir(parents=True, exist_ok=True)
    with open(pdir / "profile.json", "w") as f:
        json.dump(data, f, indent=4)

st.set_page_config(page_title="MyoGait DCM Dashboard", layout="wide")

col_title, col_logo = st.columns([6, 1])
with col_title:
    st.title("Markerless Gait Analysis Pipeline")
    st.caption("Powered by the MyoGait open-source framework.")
with col_logo:
    st.image("logo.png", use_container_width=True)

tab_pipe, tab_db = st.tabs(["🏃 Pipeline Analysis", "📂 Clinical Database (CRF)"])

# ==========================================
# TAB 1: PIPELINE ANALYSIS
# ==========================================
with tab_pipe:
    st.header("Pipeline Setup")
    
    col1, col2 = st.columns(2)
    with col1:
        experimenters = load_json_list(EXP_DB)
        exp_sel = st.selectbox("Experimenter (Pipeline Runner)", experimenters + ["+ Manage List"])
        if exp_sel == "+ Manage List":
            manage_col1, manage_col2 = st.columns(2)
            new_exp = manage_col1.text_input("New Experimenter Name")
            if manage_col1.button("Add"):
                if new_exp and new_exp not in experimenters:
                    experimenters.append(new_exp)
                    save_json_list(EXP_DB, experimenters)
                    st.success("Added! Please refresh or re-select.")
            
            del_exp = manage_col2.selectbox("Remove Experimenter", experimenters)
            if manage_col2.button("Remove"):
                if del_exp in experimenters:
                    experimenters.remove(del_exp)
                    save_json_list(EXP_DB, experimenters)
                    st.success("Removed! Please refresh or re-select.")
            exp_sel = experimenters[0] if experimenters else ""
                
    with col2:
        patients = get_patients()
        if not patients:
            st.warning("No patients found in database. Create one in the Clinical Database tab first.")
            pat_sel = None
        else:
            pat_sel = st.selectbox("Select Patient", patients)
            
    if pat_sel:
        profile = load_patient(pat_sel)
        height_val = profile.get("height_cm", 0.0) / 100.0 if profile.get("height_cm") else None
        if height_val:
            st.success(f"Loaded {pat_sel}: Height = {height_val}m")
        else:
            st.warning(f"Loaded {pat_sel}: No height recorded. Kinematics will not be scaled.")
            
        st.markdown("---")
        
        # Pipeline Settings
        col3, col4 = st.columns(2)
        with col3:
            from myogait.events import EVENT_METHODS
            from myogait.models import list_models
            
            plane = st.selectbox("Recording Plane", ["sagittal", "coronal"], index=0)
            
            available_models = list_models()
            model = st.selectbox("Pose Model", available_models, index=available_models.index("mediapipe") if "mediapipe" in available_models else 0)
            
            cutoff_hz = st.slider("Low-pass Filter Cutoff (Hz)", 1.0, 10.0, 5.0, 0.5)
            events_method = st.selectbox("Gait Event Method", list(EVENT_METHODS.keys()), index=0, help="Standard: zeni, crossing, velocity, oconnor.\nAdvanced (gk_): Deep learning and literature models (e.g. gk_ensemble for multi-model voting).")
            
            quality_preset_help = (
                "Standard: Stricter gating — requires each leg landmark to be clearly visible and well inside the frame. "
                "Use for close, well-framed recordings where the subject fills a good portion of the image. Matches upstream MyoGait defaults.\n"
                "Distant subject: Looser gating plus an adaptive floor that scales the visibility threshold to the clip's own median leg confidence. "
                "Use when the subject is far from the camera and pose confidence is globally low, which would otherwise cause usable frames to be discarded at the start and end of the clip.\n"
                "Caution: Looser gating admits lower-confidence frames, which can produce less reliable gait events near the beginning and end of a recording."
            )
            quality_preset = st.selectbox(
                "Frame Quality Preset",
                list(FRAME_QUALITY_PRESETS.keys()),
                index=0,
                help=quality_preset_help,
            )
            preset_params = FRAME_QUALITY_PRESETS[quality_preset]
            
        with col4:
            export_pdf = st.checkbox("Generate Clinical PDF Report", True, help="Generates a comprehensive clinical PDF report with summary metrics, kinematic plots, and gait cycle comparisons.")
            export_excel = st.checkbox("Generate Excel Workbook", True, help="Generates a multi-sheet Excel workbook containing raw angles, spatiotemporal metrics, and computed clinical scores.")
            export_csv = st.checkbox("Export Raw CSV Data", True, help="Exports the raw and filtered joint kinematics data as a standard CSV file.")
            export_opensim = st.checkbox("Generate OpenSim Kinematics (.mot/.trc)", True, help="Converts the 2D kinematic data into .mot and .trc files for OpenSim musculoskeletal modeling.")
            apply_bias = st.checkbox("Apply Empirical Bias Corrections", False, help="Applies a predefined angular correction to compensate for known systematic tracking biases (e.g., knee hyperextension bias in some models).")
            
            visible_side = st.radio("Visible Side (Sagittal Only)", ["both", "left", "right"], index=0, help="When recording from the side (sagittal), the camera only clearly sees one leg. Selecting the visible side will scrub the occluded leg's data to prevent noisy tracking from skewing the clinical scores. Selecting 'both' will keep all data.")
            enable_bilateral = (visible_side == "both")

        st.markdown("---")
        st.subheader("Video Selection")
        vid_source = st.radio("Source", ["File Upload", "Local File Path"])
        
        video_path = None
        is_json = False
        
        if vid_source == "File Upload":
            uploaded_file = st.file_uploader("Upload Video (MP4/MOV) OR Data (JSON)", type=["mp4", "mov", "avi", "json"])
            if uploaded_file:
                is_json = uploaded_file.name.endswith(".json")
                temp_dir = OUT_DIR / "dashboard_uploads"
                temp_dir.mkdir(parents=True, exist_ok=True)
                tmp_file = temp_dir / uploaded_file.name
                with open(tmp_file, "wb") as f:
                    f.write(uploaded_file.read())
                video_path = str(tmp_file)
        else:
            local_path = st.text_input("Absolute path to Video or JSON file (e.g. /Users/name/video.mp4)")
            if local_path and os.path.exists(local_path):
                video_path = local_path
                is_json = local_path.endswith(".json")
            elif local_path:
                st.error("File not found.")

        if video_path:
            if st.button("Run Clinical Analysis", type="primary"):
                st.session_state["run_pipeline"] = True
                st.session_state["pipeline_results"] = None

            if st.session_state.get("run_pipeline"):
                if st.session_state.get("pipeline_results") is None:
                    with st.spinner("Processing data and computing biomechanics..."):
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        vid_name = Path(video_path).stem
                        run_out_dir = DB_DIR / pat_sel / f"{timestamp}_{vid_name}"
                        run_out_dir.mkdir(parents=True, exist_ok=True)
                        
                        try:
                            results = run_dcm_pipeline(
                                video_path=None if is_json else video_path,
                                input_json_path=video_path if is_json else None,
                                output_dir=str(run_out_dir),
                                model=model, cutoff_hz=cutoff_hz, plane=plane,
                                subject_id=pat_sel, subject_height_m=height_val,
                                events_method=events_method, apply_bias_correction=apply_bias,
                                visible_side=visible_side, enable_bilateral=enable_bilateral,
                                experimenter=exp_sel, generate_pdf=export_pdf,
                                generate_excel=export_excel, generate_csv=export_csv,
                                generate_opensim=export_opensim, generate_plots=True,
                                edge_margin=preset_params["edge_margin"],
                                min_leg_vis=preset_params["min_leg_vis"],
                                adaptive_leg_vis=preset_params["adaptive_leg_vis"],
                            )
                            st.session_state["pipeline_results"] = results
                            st.success("Analysis Complete!")
                        except Exception as e:
                            st.error(f"Pipeline failed: {e}")

                results = st.session_state.get("pipeline_results")
                if results is not None:
                    try:
                        run_out_dir = Path(results["output_dir"])
                        stats = results["stats"]
                        st_stats = stats.get("spatiotemporal", {})
                        sym_stats = stats.get("symmetry", {})
                        gps_score = stats.get("gps", {})
                        gps_val = gps_score.get("gps_2d_overall", "N/A") if isinstance(gps_score, dict) else gps_score
                        sdi_score = stats.get("sdi", "N/A")
                        sdi_val = sdi_score.get("gdi_2d_overall", "N/A") if isinstance(sdi_score, dict) else sdi_score
                        flags = stats.get("pathology_flags", [])

                        # Metrics
                        st.header("1. Clinical Summary Metrics")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Cadence (steps/min)", round(st_stats.get("cadence_steps_per_min", 0), 1) if st_stats.get("cadence_steps_per_min") else "N/A")
                        c2.metric("Symmetry Index (SI %)", round(sym_stats.get("overall_si", 0), 1) if sym_stats.get("overall_si") else "N/A")
                        c3.metric("Gait Profile Score (GPS °)", round(gps_val, 1) if isinstance(gps_val, (int, float)) else gps_val)
                        c4.metric("Sagittal Deviation (SDI)", round(sdi_val, 1) if isinstance(sdi_val, (int, float)) else sdi_val)
                        
                        if flags: st.error(f"⚠️ **Clinical Flags Detected:** {', '.join(flags)}")
                        else: st.success("✅ No extreme pathological flags detected.")

                        # Frame-Quality Thresholds Display
                        try:
                            meta_path = run_out_dir / "run_metadata.json"
                            if meta_path.exists():
                                with open(meta_path, "r", encoding="utf-8") as f_meta:
                                    run_meta = json.load(f_meta)
                                
                                eff_vis = run_meta.get("effective_min_leg_vis", run_meta.get("settings", {}).get("effective_min_leg_vis"))
                                min_vis = run_meta.get("min_leg_vis", run_meta.get("settings", {}).get("min_leg_vis"))
                                edge_m = run_meta.get("edge_margin", run_meta.get("settings", {}).get("edge_margin"))
                                adapt_vis = run_meta.get("adaptive_leg_vis", run_meta.get("settings", {}).get("adaptive_leg_vis"))

                                items = []
                                if eff_vis is not None:
                                    items.append(f"**Effective Visibility Threshold:** `{eff_vis:.3f}`" if isinstance(eff_vis, float) else f"**Effective Visibility Threshold:** `{eff_vis}`")
                                if min_vis is not None:
                                    items.append(f"**Floor (`min_leg_vis`):** `{min_vis}`")
                                if edge_m is not None:
                                    items.append(f"**Edge Margin:** `{edge_m}`")
                                if adapt_vis is not None:
                                    items.append(f"**Adaptive Floor:** `{'Enabled' if adapt_vis else 'Disabled'}`")

                                if items:
                                    st.caption(f"⚙️ **Frame Quality:** {' | '.join(items)}")
                                    if adapt_vis:
                                        st.caption("When the adaptive floor is enabled, the effective threshold is computed per recording: a value above the floor means the clip tracked well, a value at the floor means the clip was distant or low-confidence and the threshold clamped.")
                        except Exception:
                            pass

                        # Plots
                        st.header("2. Biomechanical Plots")
                        t1, t2, t3 = st.tabs(["Summary Dashboard", "Angle Timeseries", "Gait Cycles"])
                        
                        with t1:
                            sum_img = run_out_dir / "plots" / "summary.png"
                            if sum_img.exists(): st.image(str(sum_img), use_container_width=True)
                        with t2:
                            ang_img = run_out_dir / "plots" / "angles.png"
                            if ang_img.exists(): st.image(str(ang_img), use_container_width=True)
                        with t3:
                            cl, cr = st.columns(2)
                            cyc_l = run_out_dir / "plots" / "cycles_left.png"
                            cyc_r = run_out_dir / "plots" / "cycles_right.png"
                            skipped_left = None
                            skipped_right = None
                            try:
                                with open(run_out_dir / "run_metadata.json", "r", encoding="utf-8") as f:
                                    meta = json.load(f)
                                skipped = meta.get("skipped_sides", [])
                                for s in skipped:
                                    if s.get("side") == "left":
                                        skipped_left = s
                                    elif s.get("side") == "right":
                                        skipped_right = s
                            except Exception:
                                pass

                            with cl:
                                if visible_side in ["both", "left"]:
                                    if cyc_l.exists():
                                        st.image(str(cyc_l), caption="Left Cycles")
                                    elif skipped_left:
                                        st.info("Left cycles unavailable — insufficient heel-strike events detected in this recording.")
                                    else:
                                        st.info("No cycle data available for this side.")
                            with cr:
                                if visible_side in ["both", "right"]:
                                    if cyc_r.exists():
                                        st.image(str(cyc_r), caption="Right Cycles")
                                    elif skipped_right:
                                        st.info("Right cycles unavailable — insufficient heel-strike events detected in this recording.")
                                    else:
                                        st.info("No cycle data available for this side.")

                        # Video
                        st.header("3. Skeleton Overlay Video")
                        skel_vid = run_out_dir / "plots" / "skeleton_overlay.mp4"
                        if skel_vid.exists():
                            skel_vid_h264 = run_out_dir / "plots" / "skeleton_overlay_h264.mp4"
                            if not skel_vid_h264.exists():
                                subprocess.run(["ffmpeg", "-y", "-i", str(skel_vid), "-vcodec", "libx264", str(skel_vid_h264)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            if skel_vid_h264.exists(): st.video(str(skel_vid_h264))
                            else: st.video(str(skel_vid))

                        # Downloads
                        st.header("4. Downloads")
                        pdf_path = run_out_dir / "report" / "dcm_gait_report.pdf"
                        if pdf_path.exists():
                            with open(pdf_path, "rb") as f:
                                st.download_button("📄 Download Clinical PDF Report", f, file_name=f"{pat_sel}_gait_report.pdf", type="primary")

                    except Exception as e:
                        st.error(f"Error rendering results: {e}")

# ==========================================
# TAB 2: CLINICAL DATABASE
# ==========================================
with tab_db:
    st.header("Patient Clinical Database (CRF)")
    
    db_mode = st.radio("Mode", ["Select Existing Patient", "Create New Patient"])
    
    if db_mode == "Create New Patient":
        new_id = st.text_input("New Patient ID (e.g., DCM-001)")
        if st.button("Create Profile") and new_id:
            if (DB_DIR / new_id).exists():
                st.error("Patient already exists!")
            else:
                save_patient(new_id, {})
                st.success(f"Created {new_id}! Please switch to 'Select Existing Patient' to edit.")
    else:
        patients = get_patients()
        if not patients:
            st.warning("No patients in database.")
        else:
            col_sel, col_del = st.columns([3, 1])
            with col_sel:
                sel_pat = st.selectbox("Select Patient to Edit", patients)
            with col_del:
                st.write("") # spacing
                st.write("")
                if st.button("🗑️ Delete Patient", type="primary"):
                    import shutil
                    shutil.rmtree(DB_DIR / sel_pat)
                    st.success(f"Deleted {sel_pat}! Please refresh.")
                    st.stop()
            prof = load_patient(sel_pat)
            
            with st.form("crf_form"):
                import pandas as pd
                st.subheader("Section 1 — Eligibility and Enrolment")
                c_a, c_b = st.columns(2)
                group = c_a.selectbox("Group", ["DCM", "Healthy Control"], index=["DCM", "Healthy Control"].index(prof.get("group", "DCM")) if prof.get("group") in ["DCM", "Healthy Control"] else 0)
                v_date = c_b.text_input("Visit Date", prof.get("v_date", ""))
                
                c1, c2 = st.columns(2)
                d_consent = c1.text_input("Date of informed consent", prof.get("d_consent", ""))
                d_elig = c2.text_input("Date of eligibility check", prof.get("d_elig", ""))
                meets_crit = c1.radio("Participant meets all criteria?", ["Yes", "No"], index=["Yes", "No"].index(prof.get("meets_crit")) if prof.get("meets_crit") in ["Yes", "No"] else None)
                enrolled = c2.radio("Participant enrolled in study?", ["Yes", "No"], index=["Yes", "No"].index(prof.get("enrolled")) if prof.get("enrolled") in ["Yes", "No"] else None)
                
                st.subheader("Section 2 — Demographic Data")
                c3, c4 = st.columns(2)
                yob = c3.text_input("Year of birth (yyyy)", prof.get("yob", ""))
                sex = c4.selectbox("Sex", ["", "Male", "Female", "Other / prefer not to say"], index=["", "Male", "Female", "Other / prefer not to say"].index(prof.get("sex", "")) if prof.get("sex") in ["", "Male", "Female", "Other / prefer not to say"] else 0)
                h_cm = c3.number_input("Height (cm)", min_value=0.0, value=float(prof.get("height_cm", 0.0)), step=1.0)
                w_kg = c4.number_input("Weight (kg)", min_value=0.0, value=float(prof.get("weight_kg", 0.0)), step=1.0)
                if h_cm > 0 and w_kg > 0:
                    st.write(f"**Calculated BMI:** {w_kg / ((h_cm/100)**2):.1f} kg/m²")
                smoking = st.selectbox("Smoking history", ["", "Never smoked", "Former smoker", "Current smoker"], index=["", "Never smoked", "Former smoker", "Current smoker"].index(prof.get("smoking", "")) if prof.get("smoking") in ["", "Never smoked", "Former smoker", "Current smoker"] else 0)
                
                st.subheader("Section 3 — Clinical History")
                c5, c6 = st.columns(2)
                surg = c5.selectbox("Surgical status", ["", "Pre-operative", "Post-operative", "Not surgical"], index=["", "Pre-operative", "Post-operative", "Not surgical"].index(prof.get("surg", "")) if prof.get("surg") in ["", "Pre-operative", "Post-operative", "Not surgical"] else 0)
                d_surg = c6.text_input("Date of surgery", prof.get("d_surg", ""))
                lvl_decomp = c5.text_input("Levels decompressed", prof.get("lvl_decomp", ""))
                lvl_comp = c6.text_input("Levels with cord compression", prof.get("lvl_comp", ""))
                
                st.subheader("Section 4 — Clinical Assessment (mJOA & 10MWT)")
                c7, c8 = st.columns(2)
                ul_motor = c7.number_input("Upper Limb Motor (0-5)", 0, 5, int(prof.get("ul_motor", 5)))
                ll_motor = c8.number_input("Lower Limb Motor (0-7)", 0, 7, int(prof.get("ll_motor", 7)))
                ul_sensory = c7.number_input("Upper Limb Sensory (0-3)", 0, 3, int(prof.get("ul_sensory", 3)))
                bladder = c8.number_input("Bladder (0-3)", 0, 3, int(prof.get("bladder", 3)))
                
                mjoa_tot = ul_motor + ll_motor + ul_sensory + bladder
                sev = "Mild (15-17)" if mjoa_tot >= 15 else "Moderate (12-14)" if mjoa_tot >= 12 else "Severe (≤11)"
                st.write(f"**TOTAL mJOA Score:** {mjoa_tot} ({sev})")
                
                st.markdown("---")
                st.subheader("Section 4B — 10-Metre Walk Test (10MWT)")
                t10_aid = st.selectbox("Assistive device during 10MWT", ["None", "Cane", "Walker", "Other"], index=["None", "Cane", "Walker", "Other"].index(prof.get("t10_aid", "None")) if prof.get("t10_aid") in ["None", "Cane", "Walker", "Other"] else 0)
                
                st.write("**Preferred (comfortable) speed:**")
                default_pref = [{"Trial 1 (s)": "", "Trial 2 (s)": "", "Average time (s)": "", "Average velocity (m/s)": ""}]
                pref_df = pd.DataFrame(prof.get("t10_pref", default_pref))
                ed_pref = st.data_editor(pref_df, use_container_width=True, hide_index=True, key="pref_grid")
                
                st.write("**Fast (maximum safe) speed:**")
                default_fast = [{"Trial 1 (s)": "", "Trial 2 (s)": "", "Average time (s)": "", "Average velocity (m/s)": ""}]
                fast_df = pd.DataFrame(prof.get("t10_fast", default_fast))
                ed_fast = st.data_editor(fast_df, use_container_width=True, hide_index=True, key="fast_grid")
                
                t10_comments = st.text_input("10MWT comments", prof.get("t10_comments", ""))
                t10_ae = st.radio("Adverse event during 10MWT?", ["No", "Yes"], index=0 if prof.get("t10_ae", "No") == "No" else 1)
                t10_ae_desc = ""
                if t10_ae == "Yes":
                    t10_ae_desc = st.text_input("Please describe the adverse event", prof.get("t10_ae_desc", ""))
                
                st.subheader("Section 5 — Gait Assessment Setup")
                assessors = load_json_list(ASSESSOR_DB)
                c9, c10 = st.columns(2)
                
                current_assessor = prof.get("assessor", "")
                if current_assessor and current_assessor not in assessors:
                    assessors.append(current_assessor)
                    
                assessor = c9.selectbox("Assessor", assessors, index=assessors.index(current_assessor) if current_assessor in assessors else 0)
                
                with c10.expander("🛠️ Add/Remove Assessor"):
                    new_a = st.text_input("New Assessor Name")
                    if st.form_submit_button("Add to List") and new_a:
                        if new_a not in assessors:
                            assessors.append(new_a)
                            save_json_list(ASSESSOR_DB, assessors)
                            st.rerun()
                    
                    del_a = st.selectbox("Remove Assessor", assessors)
                    if st.form_submit_button("Remove from List") and del_a:
                        if del_a in assessors:
                            assessors.remove(del_a)
                            save_json_list(ASSESSOR_DB, assessors)
                            st.rerun()
                
                gait_aid = c9.selectbox("Usual gait aid", ["", "None — independent", "Single cane", "Walker / rollator", "Crutches", "Other"], index=["", "None — independent", "Single cane", "Walker / rollator", "Crutches", "Other"].index(prof.get("gait_aid", "")) if prof.get("gait_aid") in ["", "None — independent", "Single cane", "Walker / rollator", "Crutches", "Other"] else 0)
                tape = c10.radio("Tape applied to joints?", ["No", "Yes"], index=0 if prof.get("tape", "No") == "No" else 1)
                
                st.subheader("Section 5B & 5D — Walking Trials & Video Log")
                default_log = [{"Trial": i, "Direction": "", "Time (s)": "", "Speed (m/s)": "", "Sagittal File Path": "", "Coronal File Path": ""} for i in range(1, 9)]
                video_log_df = pd.DataFrame(prof.get("video_log", default_log))
                edited_log_df = st.data_editor(video_log_df, use_container_width=True, hide_index=True)
                
                if st.form_submit_button("💾 Save Patient Profile"):
                    data = {
                        "group": group, "v_date": v_date,
                        "d_consent": d_consent, "d_elig": d_elig, "meets_crit": meets_crit, "enrolled": enrolled,
                        "yob": yob, "sex": sex, "height_cm": h_cm, "weight_kg": w_kg, "smoking": smoking,
                        "surg": surg, "d_surg": d_surg, "lvl_decomp": lvl_decomp, "lvl_comp": lvl_comp,
                        "ul_motor": ul_motor, "ll_motor": ll_motor, "ul_sensory": ul_sensory, "bladder": bladder,
                        "t10_aid": t10_aid, "t10_pref": ed_pref.to_dict("records"), "t10_fast": ed_fast.to_dict("records"),
                        "t10_comments": t10_comments, "t10_ae": t10_ae, "t10_ae_desc": t10_ae_desc,
                        "assessor": assessor, "gait_aid": gait_aid, "tape": tape,
                        "video_log": edited_log_df.to_dict("records")
                    }
                    save_patient(sel_pat, data)
                    st.success("Profile saved successfully! Please refresh or switch tabs to use new height.")
