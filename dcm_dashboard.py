import os
import tempfile
from pathlib import Path

# Configure cache, model, and temp directories on workspace drive to prevent C: drive exhaustion
BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / ".tmp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(TEMP_DIR)
os.environ["TMP"] = str(TEMP_DIR)
os.environ["TMPDIR"] = str(TEMP_DIR)
tempfile.tempdir = str(TEMP_DIR)

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = BASE_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MYOGAIT_MODELS_DIR"] = str(MODELS_DIR)
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import streamlit as st
import json
import datetime

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

from dcm_pipeline import (  # noqa: E402
    FRAME_QUALITY_PRESETS, GROUPS, SETTINGS, SETTINGS_LIST,
    calculation_deviations, default_settings, resolve_settings, skipped_message,
)

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

def get_hardware_info():
    """Detect GPU hardware, VRAM, and return recommendations and waivers."""
    info = {
        "has_torch": False,
        "cuda_available": False,
        "device_name": "CPU (Lightweight / Standard)",
        "vram_gb": 0.0,
        "tier": "cpu",
        "badge": "💻 CPU Mode",
        "recommended": ["mediapipe", "yolo"],
        "waiver": "Running on CPU. MediaPipe is strongly recommended for fast inference. Sapiens / ViTPose may be slow without a dedicated GPU."
    }
    try:
        import torch
        info["has_torch"] = True
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["device_name"] = torch.cuda.get_device_name(0)
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            info["vram_gb"] = round(total_bytes / (1024 ** 3), 1)

            if info["vram_gb"] >= 16.0:
                info["tier"] = "ultra"
                info["badge"] = f"🔥 Workstation GPU: {info['device_name']} ({info['vram_gb']} GB VRAM)"
                info["recommended"] = ["sapiens2-ultra", "sapiens2-top", "vitpose-huge", "sapiens-top", "yolo", "mediapipe"]
                info["waiver"] = "✅ Full Clearance: High VRAM (≥16GB) detected. All models, including Sapiens 2 Ultra (5B), Sapiens 2 Top (1B), and ViTPose Huge, can run at maximum accuracy without OOM risk."
            elif info["vram_gb"] >= 8.0:
                info["tier"] = "high"
                info["badge"] = f"⚡ High-End GPU: {info['device_name']} ({info['vram_gb']} GB VRAM)"
                info["recommended"] = ["sapiens2-mid", "sapiens2-quick", "sapiens-mid", "vitpose-large", "yolo", "mediapipe"]
                info["waiver"] = "✅ High VRAM (8-12GB). Sapiens 2 Mid (0.8B) and ViTPose Large recommended. Sapiens 2 Ultra (5B) may exceed memory on long 4K clips."
            else:
                info["tier"] = "entry"
                info["badge"] = f"⚡ Entry/Laptop GPU: {info['device_name']} ({info['vram_gb']} GB VRAM)"
                info["recommended"] = ["sapiens2-quick", "vitpose", "yolo", "rtmw", "mediapipe"]
                info["waiver"] = "⚠️ Entry VRAM (<8GB). Recommended: Sapiens 2 Quick (0.4B), ViTPose Base, YOLO, or MediaPipe. Avoid 1B+ / 5B models to prevent CUDA Out-of-Memory."
    except Exception:
        pass
    return info

# -----------------
# Analysis settings panel (built from dcm_pipeline.SETTINGS_LIST)
# -----------------
def _cfg_key(key):
    return f"cfg_{key}"


def _matching_preset():
    for name, values in FRAME_QUALITY_PRESETS.items():
        if all(st.session_state.get(_cfg_key(k)) == v for k, v in values.items()):
            return name
    return "Custom"


def _apply_preset():
    name = st.session_state.get("cfg_frame_preset")
    for k, v in FRAME_QUALITY_PRESETS.get(name, {}).items():
        st.session_state[_cfg_key(k)] = v


def _sync_preset():
    st.session_state["cfg_frame_preset"] = _matching_preset()


def _reset_settings():
    for key, value in default_settings().items():
        st.session_state[_cfg_key(key)] = value
    _sync_preset()


def _init_settings_state():
    for key, value in default_settings().items():
        st.session_state.setdefault(_cfg_key(key), value)
    st.session_state.setdefault("cfg_frame_preset", _matching_preset())


def _setting_widget(spec, hw_info):
    key = _cfg_key(spec.key)
    helptext = f"{spec.help}\n\nDefault: {spec.default!r}  ·  Source: {spec.source}"
    on_change = _sync_preset if spec.key in ("edge_margin", "min_leg_vis", "adaptive_leg_vis") else None
    if spec.kind == "bool":
        st.checkbox(spec.label, key=key, help=helptext, on_change=on_change)
    elif spec.kind in ("float", "int"):
        cast = float if spec.kind == "float" else int
        st.number_input(spec.label, min_value=cast(spec.min), max_value=cast(spec.max),
                        step=cast(spec.step), key=key, help=helptext, on_change=on_change,
                        format="%.3f" if spec.kind == "float" else "%d")
    elif spec.kind == "model":
        from myogait.models import list_models
        options = list_models()
        if st.session_state[key] not in options:
            options = [st.session_state[key]] + options
        st.selectbox(spec.label, options, key=key, help=helptext)
        chosen = st.session_state[key]
        if ("sapiens" in chosen or "vitpose" in chosen) and not hw_info["cuda_available"]:
            st.warning("No NVIDIA GPU detected: this model will be very slow on CPU. "
                       "'mediapipe' runs on CPU (less accurate than myogait's default).")
        elif "sapiens2-ultra" in chosen and hw_info["vram_gb"] < 16.0:
            st.warning("Sapiens 2 Ultra (5B) needs ≥16 GB of GPU memory; use sapiens2-mid or sapiens2-quick.")
    elif spec.kind == "choice":
        if spec.extras.get("choices_from") == "myogait.events.EVENT_METHODS":
            from myogait.events import EVENT_METHODS
            options = list(EVENT_METHODS.keys())
        else:
            options = list(spec.choices)
        st.selectbox(spec.label, options, key=key, help=helptext)


def render_settings_panel(hw_info):
    """Draw every analysis setting with its explanation; return the values."""
    _init_settings_state()
    current = {k: st.session_state[_cfg_key(k)] for k in SETTINGS}
    devs = calculation_deviations(current)

    st.subheader("Analysis Settings")
    head_l, head_r = st.columns([4, 1])
    with head_l:
        if devs:
            st.warning("**Changed from myogait defaults (these change the numbers):** "
                       + ", ".join(f"{d['label']} = `{d['value']}` (default `{d['default']}`)" for d in devs))
        else:
            st.success("✅ All calculation settings are myogait defaults — results are identical "
                       "to `myogait.run_pipeline`. Hover the ⓘ next to any setting for an explanation.")
    with head_r:
        st.button("↺ Reset to myogait defaults", on_click=_reset_settings, use_container_width=True)

    for group in GROUPS:
        specs = [s for s in SETTINGS_LIST if s.group == group]
        n_changed = sum(1 for s in specs if st.session_state[_cfg_key(s.key)] != s.default)
        title = f"{group}" + (f"  —  {n_changed} changed" if n_changed else "")
        with st.expander(title, expanded=False):
            if group == "Frame quality":
                st.selectbox("Preset", list(FRAME_QUALITY_PRESETS) + ["Custom"], key="cfg_frame_preset",
                             on_change=_apply_preset,
                             help="Fills the three settings below. 'Standard' is myogait's behaviour; "
                                  "'Distant subject' keeps more frames when the patient is far from "
                                  "the camera and pose confidence is low.")
            cols = st.columns(2)
            for i, spec in enumerate(specs):
                with cols[i % 2]:
                    _setting_widget(spec, hw_info)

    return resolve_settings({k: st.session_state[_cfg_key(k)] for k in SETTINGS})
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

        # Hardware & Compute Status
        hw_info = get_hardware_info()
        with st.expander(f"🖥️ Hardware & GPU Status: {hw_info['badge']}", expanded=False):
            st.markdown(f"**Compute Device:** `{hw_info['device_name']}`" + (f" (`{hw_info['vram_gb']} GB VRAM`)" if hw_info["cuda_available"] else ""))
            st.info(hw_info["waiver"])
            st.markdown("**Recommended Models for this System:** " + ", ".join([f"`{m}`" for m in hw_info["recommended"]]))

        # Pipeline Settings (every setting, with explanations, from dcm_pipeline.SETTINGS_LIST)
        settings = render_settings_panel(hw_info)

        with st.expander("Outputs", expanded=False):
            oc1, oc2 = st.columns(2)
            export_pdf = oc1.checkbox("Generate Clinical PDF Report", True, help="Multi-page clinical PDF report with summary metrics, kinematic plots and gait cycle comparisons.")
            export_excel = oc1.checkbox("Generate Excel Workbook", True, help="Multi-sheet Excel workbook with angles, spatiotemporal metrics and clinical scores.")
            export_csv = oc1.checkbox("Export CSV Data", True, help="Joint-angle time series, cycle curves and summary statistics as CSV files.")
            export_opensim = oc2.checkbox("Generate OpenSim Kinematics (.mot/.trc)", True, help="2D kinematics as OpenSim .mot and .trc files for musculoskeletal modelling.")
            export_videos = oc2.checkbox("Render Skeleton & Stick-Figure Videos", True, help="Skeleton overlay on the original video plus an anonymised stick-figure video. Rendering takes extra time on long clips.")

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
                                subject_id=pat_sel, subject_height_m=height_val,
                                experimenter=exp_sel, settings=settings,
                                generate_pdf=export_pdf, generate_excel=export_excel,
                                generate_csv=export_csv, generate_opensim=export_opensim,
                                generate_plots=True, generate_videos=export_videos,
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
                        run_settings = results.get("settings") or {}
                        visible_side = run_settings.get("visible_side", "both")

                        # Metrics
                        st.header("1. Clinical Summary Metrics")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Cadence (steps/min)", round(st_stats.get("cadence_steps_per_min", 0), 1) if st_stats.get("cadence_steps_per_min") else "N/A")
                        c2.metric("Symmetry Index (SI %)", round(sym_stats.get("overall_si", 0), 1) if sym_stats.get("overall_si") else "N/A")
                        c3.metric("Gait Profile Score (GPS °)", round(gps_val, 1) if isinstance(gps_val, (int, float)) else gps_val)
                        c4.metric("Sagittal Deviation (SDI)", round(sdi_val, 1) if isinstance(sdi_val, (int, float)) else sdi_val)

                        if flags: st.error(f"⚠️ **Clinical Flags Detected:** {', '.join(flags)}")
                        else: st.success("✅ No extreme pathological flags detected.")

                        run_devs = calculation_deviations(run_settings) if run_settings else []
                        if run_devs:
                            st.caption("⚙️ **Settings changed from myogait defaults for this run:** "
                                       + ", ".join(f"{d['label']} = `{d['value']}`" for d in run_devs))
                        else:
                            st.caption("⚙️ All calculation settings were myogait defaults for this run.")
                        for w in (results.get("quality") or {}).get("warnings", []):
                            st.warning(f"myogait quality check: {w}")
                        for sk in results.get("skipped_sides", []):
                            st.warning(skipped_message(sk))
                        if results.get("outputs_failed"):
                            st.error("Some outputs could not be generated: " + ", ".join(results["outputs_failed"]))

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
                                        st.info(skipped_message(skipped_left))
                                    else:
                                        st.info("No cycle data available for this side.")
                            with cr:
                                if visible_side in ["both", "right"]:
                                    if cyc_r.exists():
                                        st.image(str(cyc_r), caption="Right Cycles")
                                    elif skipped_right:
                                        st.info(skipped_message(skipped_right))
                                    else:
                                        st.info("No cycle data available for this side.")

                        # Video Visualizations
                        st.header("3. Video Visualizations")
                        tab_skel, tab_stick = st.tabs(["🎥 Skeleton Overlay Video", "👤 Anonymized Stick-Figure Video"])

                        with tab_skel:
                            skel_vid = run_out_dir / "plots" / "skeleton_overlay.mp4"
                            skel_vid_h264 = run_out_dir / "plots" / "skeleton_overlay_h264.mp4"
                            target_vid = skel_vid_h264 if skel_vid_h264.exists() else skel_vid
                            if target_vid.exists():
                                st.video(str(target_vid))
                            else:
                                st.info("Skeleton overlay video was not generated for this session.")

                        with tab_stick:
                            stick_vid = run_out_dir / "plots" / "stickfigure_video.mp4"
                            if stick_vid.exists():
                                st.video(str(stick_vid))
                                st.caption("Anonymized 2D Stick-Figure Video (De-identified for patient privacy, publications & conferences)")
                            else:
                                st.info("Anonymized stick figure video is not available for this session.")

                        # Downloads
                        st.header("4. Downloads & Clinical Exports")
                        dl_col1, dl_col2, dl_col3 = st.columns(3)

                        pdf_path = run_out_dir / "report" / "dcm_gait_report.pdf"
                        with dl_col1:
                            if pdf_path.exists():
                                with open(pdf_path, "rb") as f:
                                    st.download_button("📄 Download Clinical PDF Report", f, file_name=f"{pat_sel}_gait_report.pdf", type="primary", use_container_width=True)

                        stick_path = run_out_dir / "plots" / "stickfigure_video.mp4"
                        with dl_col2:
                            if stick_path.exists():
                                with open(stick_path, "rb") as f_stick:
                                    st.download_button("👤 Download Stick-Figure Video", f_stick, file_name=f"{pat_sel}_stickfigure.mp4", use_container_width=True)

                        xlsx_path = run_out_dir / "data" / "dcm_gait_analysis.xlsx"
                        with dl_col3:
                            if xlsx_path.exists():
                                with open(xlsx_path, "rb") as f_xlsx:
                                    st.download_button("📊 Download Excel Workbook", f_xlsx, file_name=f"{pat_sel}_analysis.xlsx", use_container_width=True)

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
