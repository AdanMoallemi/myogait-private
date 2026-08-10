#!/usr/bin/env python3
"""DCM Gait Analysis Pipeline.

A clinical gait analysis pipeline tailored for Degenerative Cervical Myelopathy (DCM)
and pathological gait evaluation using myogait.

Designed for 5m walkway recordings (sagittal and coronal/frontal views).

Key Features:
- Flexible pose model selection (Sapiens 2 / MediaPipe / YOLO / etc.)
- Spike removal via Median filter + zero-phase Butterworth low-pass filtering
- Sagittal & Coronal joint angle computation
- Zero-parameter Perspective Correction (foreshortening compensation)
- Linear Detrending (cam-to-subject distance drift correction along 5m walkway)
- Preserves pathological gait signatures (skips empirical healthy-adult bias corrections)
- Gait event detection (Zeni algorithm) & cycle segmentation
- Spatiotemporal metrics, Gait Profile Scores (GPS/GVS), Symmetry Index (SI)
- Automated organized outputs: PDF Report, Excel Workbook, CSV Tables, OpenSim (.mot & .trc), Summary PNGs,
  plus auto-generated README.md and run_metadata.json for auditing.

Usage:
    # Single video processing
    python dcm_pipeline.py --video path/to/video.mp4 --output-dir ./results/patient01

    # Single video with subject metadata
    python dcm_pipeline.py --video path/to/video.mp4 --subject-id DCM_001 --height 1.72 --output-dir ./results/DCM_001

    # Batch processing an entire folder of videos
    python dcm_pipeline.py --batch-dir ./patient_videos --output-dir ./batch_results
"""

import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for plots
import matplotlib.pyplot as plt

# Import myogait components
import myogait
from myogait import (
    extract,
    normalize,
    compute_angles,
    compute_extended_angles,
    compute_frontal_angles,
    apply_perspective_correction,
    apply_linear_detrend,
    detect_events,
    segment_cycles,
    analyze_gait,
    load_json,
    save_json,
    set_subject,
)
from myogait.scores import (
    gait_variable_scores,
    gait_profile_score_2d,
    sagittal_deviation_index,
)
from myogait.plotting import (
    plot_summary,
    plot_angles,
    plot_events,
    plot_cycles,
    plot_normative_comparison,
)
from myogait.export import (
    export_csv,
    export_excel,
    export_mot,
    export_trc,
)
from myogait.report import generate_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dcm_pipeline")


def _write_run_metadata(
    output_dir: Path,
    video_path: str,
    plane: str,
    subject_id: Optional[str],
    subject_height_m: Optional[float],
    model: str,
    cutoff_hz: float,
    events_method: str,
    stats: dict,
    elapsed_s: float,
) -> Path:
    """Save machine-readable run configuration and clinical summary JSON."""
    meta_path = output_dir / "run_metadata.json"
    st = stats.get("spatiotemporal", {})
    sym = stats.get("symmetry", {})
    gps_val = stats.get("gps", {})
    if isinstance(gps_val, dict):
        gps_overall = gps_val.get("gps_overall")
    else:
        gps_overall = gps_val

    meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "myogait_version": getattr(myogait, "__version__", "0.6.0"),
        "input": {
            "video_path": str(Path(video_path).resolve()),
            "video_filename": Path(video_path).name,
            "plane": plane,
            "subject_id": subject_id or "Unspecified",
            "subject_height_m": subject_height_m,
        },
        "settings": {
            "pose_model": model,
            "cutoff_frequency_hz": cutoff_hz,
            "filters": [
                {"type": "median", "kernel_size": 5},
                {"type": "butterworth", "cutoff": cutoff_hz},
            ],
            "perspective_correction": True,
            "linear_detrending": True,
            "empirical_healthy_bias_corrections": False,
            "events_detection_method": events_method,
        },
        "results_summary": {
            "cadence_steps_per_min": st.get("cadence_steps_per_min"),
            "stride_time_mean_s": st.get("stride_time_mean_s"),
            "stride_time_std_s": st.get("stride_time_std_s"),
            "stance_pct_left": st.get("stance_pct_left"),
            "stance_pct_right": st.get("stance_pct_right"),
            "overall_symmetry_index_pct": sym.get("overall_si"),
            "gait_profile_score_gps": gps_overall,
            "sagittal_deviation_index_sdi": stats.get("sdi"),
            "clinical_flags": stats.get("pathology_flags", []),
        },
        "performance": {
            "elapsed_seconds": round(elapsed_s, 2),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta_path


def _write_run_readme(
    output_dir: Path,
    video_path: str,
    plane: str,
    subject_id: Optional[str],
    subject_height_m: Optional[float],
    model: str,
    cutoff_hz: float,
    events_method: str,
    stats: dict,
    elapsed_s: float,
) -> Path:
    """Generate human-readable Markdown README documenting the run settings & findings."""
    readme_path = output_dir / "README.md"
    st = stats.get("spatiotemporal", {})
    sym = stats.get("symmetry", {})
    gps_val = stats.get("gps", {})
    if isinstance(gps_val, dict):
        gps_overall = gps_val.get("gps_overall")
    else:
        gps_overall = gps_val

    flags = stats.get("pathology_flags", [])
    flags_formatted = "\n".join([f"- ⚠️ **{f}**" for f in flags]) if flags else "- None detected within standard thresholds."

    content = f"""# DCM Gait Analysis Run Report

**Run Timestamp**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Subject ID**: `{subject_id or 'Unspecified'}`  
**Subject Height**: `{subject_height_m or 'Unspecified'} m`  
**Video File**: `{Path(video_path).name}`  
**Recording Plane**: `{plane.capitalize()}`  
**Pose Model**: `{model}`  
**Execution Time**: `{elapsed_s:.1f} s`  

---

## 1. Executive Clinical Summary

| Metric | Value | Reference / Notes |
| :--- | :--- | :--- |
| **Cadence** | `{st.get('cadence_steps_per_min', 'N/A')}` steps/min | Step rate |
| **Stride Time** | `{st.get('stride_time_mean_s', 'N/A')} ± {st.get('stride_time_std_s', 'N/A')}` s | Mean ± SD |
| **Left Stance Phase** | `{st.get('stance_pct_left', 'N/A')}` % | % Gait cycle |
| **Right Stance Phase** | `{st.get('stance_pct_right', 'N/A')}` % | % Gait cycle |
| **Overall Symmetry Index (SI)** | `{sym.get('overall_si', 'N/A')}` % | Normal < 10% |
| **Gait Profile Score (GPS)** | `{gps_overall}` ° | Lower = closer to normative |
| **Sagittal Deviation Index (SDI)** | `{stats.get('sdi', 'N/A')}` | Normal ~ 100 |

### Pathological Gait Flags
{flags_formatted}

---

## 2. Pipeline Settings & Processing Configuration

- **Pose Estimation Backend**: `{model}`
- **Spike Removal Filter**: 1D Median Filter (`kernel_size=5`)
- **Kinematic Denoising Filter**: Low-pass 2nd-order zero-phase Butterworth filter (`cutoff={cutoff_hz} Hz`)
- **Joint Angle Reference**: `sagittal_vertical_axis` (Davis et al. vertical reference method)
- **Perspective Correction**: `Enabled` (Zero-parameter physics-based cos α foreshortening fix)
- **Linear Detrending**: `Enabled` (Removes camera-to-subject walk-along distance drift on 5m walkway)
- **Empirical Healthy Bias Corrections**: `Bypassed` (**Preserves DCM pathological signatures**, e.g., foot drop, stiff knee)
- **Gait Event Detection**: `{events_method}` algorithm (Zeni et al. foot-to-pelvis displacement/velocity thresholding)

---

## 3. Output Folder Structure

```text
{output_dir.name}/
├── README.md                      <- This clinical & technical run summary
├── run_metadata.json              <- Machine-readable exact run configuration & stats
├── report/
│   └── dcm_gait_report.pdf        <- Multi-page Clinical PDF Report
├── data/
│   ├── dcm_gait_data.json         <- Full raw/normalized pivot JSON landmark database
│   ├── dcm_gait_analysis.xlsx     <- Multi-tab Excel workbook with stats & cycle curves
│   └── csv/                       <- Raw & cycle time-series CSV data tables
│       ├── raw_angles.csv
│       ├── cycles_left.csv
│       ├── cycles_right.csv
│       └── summary_stats.csv
├── opensim/
│   ├── kinematics.mot             <- OpenSim joint angle kinematics (.mot format)
│   └── markers.trc                <- OpenSim 3D marker trajectory data (.trc format)
└── plots/
    ├── summary.png                <- Full clinical summary dashboard
    ├── angles.png                 <- Joint angle time-series plot
    ├── events.png                 <- Event detection plot (heel-strike & toe-off)
    ├── cycles_left.png            <- Normalized left gait cycle curves (0-100%)
    └── cycles_right.png           <- Normalized right gait cycle curves (0-100%)
```

---
*Generated automatically by myogait DCM Clinical Gait Analysis Pipeline v{getattr(myogait, '__version__', '0.6.0')}.*
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    return readme_path


def run_dcm_pipeline(
    video_path: str,
    output_dir: str,
    model: str = "mediapipe",
    cutoff_hz: float = 5.0,
    plane: str = "sagittal",
    subject_id: Optional[str] = None,
    subject_height_m: Optional[float] = None,
    events_method: str = "zeni",
    language: str = "en",
    generate_pdf: bool = True,
    generate_excel: bool = True,
    generate_csv: bool = True,
    generate_opensim: bool = True,
    generate_plots: bool = True,
) -> Dict[str, Any]:
    """Execute the customized DCM gait analysis pipeline on a single video file.

    Parameters
    ----------
    video_path : str
        Path to the sagittal or coronal walking video.
    output_dir : str
        Directory where reports and export files will be written.
    model : str
        Pose estimation backend ('mediapipe', 'sapiens2', 'sapiens-0.3b', etc.).
    cutoff_hz : float
        Low-pass Butterworth filter cutoff frequency (default 5.0 Hz).
    plane : str
        Recording plane ('sagittal' or 'coronal').
    subject_id : str, optional
        Subject identifier (e.g. 'DCM_001').
    subject_height_m : float, optional
        Subject height in meters for OpenSim scaling (e.g. 1.75).
    events_method : str
        Algorithm for gait event detection ('zeni', 'velocity', etc.).
    generate_pdf : bool
        Whether to produce a multi-page clinical PDF report.
    generate_excel : bool
        Whether to export an Excel workbook (.xlsx).
    generate_csv : bool
        Whether to export CSV time-series and cycle files.
    generate_opensim : bool
        Whether to export OpenSim .mot and .trc files.
    generate_plots : bool
        Whether to generate PNG plot figures.

    Returns
    -------
    dict
        Dictionary containing pipeline data, cycle segmentations, and statistics.
    """
    video_p = Path(video_path)
    if not video_p.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    # Build automatic descriptive run folder: <subject_or_video>_<plane>_<model>_<YYYYMMDD_HHMMSS>
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    subject_slug = subject_id or video_p.stem
    folder_name = f"{subject_slug}_{plane}_{model}_{timestamp_str}"

    base_out = Path(output_dir) if output_dir else Path("./Output")
    out_p = base_out / folder_name
    out_p.mkdir(parents=True, exist_ok=True)

    # Subdirectories for clean organization
    plots_p = out_p / "plots"
    report_p = out_p / "report"
    data_p = out_p / "data"
    csv_p = data_p / "csv"
    opensim_p = out_p / "opensim"

    plots_p.mkdir(parents=True, exist_ok=True)
    report_p.mkdir(parents=True, exist_ok=True)
    data_p.mkdir(parents=True, exist_ok=True)
    csv_p.mkdir(parents=True, exist_ok=True)
    opensim_p.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    logger.info("=" * 60)
    logger.info(f"Starting DCM Pipeline for: {video_p.name}")
    logger.info(f"Pose Model: {model} | Plane: {plane} | Filter Cutoff: {cutoff_hz} Hz")
    logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Pose Extraction
    # -------------------------------------------------------------------------
    logger.info("[1/7] Extracting pose landmarks...")
    t0 = time.time()
    data = extract(
        str(video_p),
        model=model,
        with_depth=False,  # Pure 2D pose extraction (avoids unauthenticated depth repo download)
    )
    if "meta" not in data:
        data["meta"] = {}
    data["meta"]["plane"] = plane

    n_frames = len(data.get("frames", []))
    detected = sum(1 for f in data.get("frames", []) if f.get("confidence", 0) > 0.3)
    logger.info(f"      Extracted {n_frames} frames ({detected} detected, {100*detected/max(n_frames, 1):.1f}%) in {time.time()-t0:.1f}s")

    # Set subject metadata if provided
    if subject_id or subject_height_m:
        data = set_subject(
            data,
            subject_id=subject_id or "DCM_Subject",
            height_m=subject_height_m,
        )

    # -------------------------------------------------------------------------
    # 2. Filtering & Signal Cleaning
    # -------------------------------------------------------------------------
    logger.info("[2/7] Filtering pose data (Median spike removal + 5Hz Butterworth)...")
    data = normalize(
        data,
        steps=[
            {"type": "median", "kernel_size": 5},
            {"type": "butterworth", "cutoff": cutoff_hz},
        ],
    )

    # -------------------------------------------------------------------------
    # 3. Angle Computation (Sagittal & Frontal)
    # -------------------------------------------------------------------------
    logger.info("[3/7] Computing joint angles...")
    data = compute_angles(data, method="sagittal_vertical_axis")
    data = compute_extended_angles(data)
    data = compute_frontal_angles(data)

    # -------------------------------------------------------------------------
    # 4. Geometry & Perspective Corrections (Preserving DCM Pathology)
    # -------------------------------------------------------------------------
    logger.info("[4/7] Applying Perspective Correction & Linear Detrending...")
    data = apply_perspective_correction(data)
    data = apply_linear_detrend(data)
    # NOTE: Empirical healthy-adult bias corrections are explicitly bypassed.

    # -------------------------------------------------------------------------
    # 5. Gait Event Detection & Cycle Segmentation
    # -------------------------------------------------------------------------
    logger.info(f"[5/7] Detecting gait events ({events_method}) and segmenting cycles...")
    data = detect_events(data, method=events_method)
    cycles_result = segment_cycles(data)
    n_cycles = len(cycles_result.get("cycles", []))
    logger.info(f"      Identified {n_cycles} valid gait cycles.")

    # -------------------------------------------------------------------------
    # 6. Biomechanical & Clinical Pathology Analysis
    # -------------------------------------------------------------------------
    logger.info("[6/7] Analyzing spatiotemporal metrics, symmetry, & gait scores...")
    stats = analyze_gait(data, cycles_result)

    # Compute Gait Profile Scores (GPS / GVS)
    gvs = gait_variable_scores(cycles_result)
    gps_score = gait_profile_score_2d(cycles_result)
    sdi_score = sagittal_deviation_index(cycles_result)

    stats["gvs"] = gvs
    stats["gps"] = gps_score
    stats["sdi"] = sdi_score

    # Log summary statistics
    st = stats.get("spatiotemporal", {})
    sym = stats.get("symmetry", {})
    logger.info(f"      Cadence: {st.get('cadence_steps_per_min', 'N/A')} steps/min")
    logger.info(f"      Stride Time: {st.get('stride_time_mean_s', 'N/A')} +/- {st.get('stride_time_std_s', 'N/A')} s")
    logger.info(f"      Overall Symmetry Index: {sym.get('overall_si', 'N/A')}%")
    logger.info(f"      Gait Profile Score (GPS): {gps_score.get('gps_overall', 'N/A') if isinstance(gps_score, dict) else gps_score}")

    flags = stats.get("pathology_flags", [])
    if flags:
        for flag in flags:
            logger.info(f"      ⚠️ Clinical Flag: {flag}")

    # Save primary JSON inside data/
    json_path = data_p / "dcm_gait_data.json"
    save_json(data, str(json_path))

    # -------------------------------------------------------------------------
    # 7. Organized Exports & Clinical Reports
    # -------------------------------------------------------------------------
    logger.info("[7/7] Generating clinical outputs and exports...")

    # A. Figures / Plots inside plots/
    if generate_plots:
        try:
            fig = plot_summary(data, cycles_result, stats)
            fig.savefig(plots_p / "summary.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            fig = plot_angles(data)
            fig.savefig(plots_p / "angles.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            fig = plot_events(data)
            fig.savefig(plots_p / "events.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            for side in ("left", "right"):
                fig = plot_cycles(cycles_result, side=side)
                fig.savefig(plots_p / f"cycles_{side}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)

            # Render color-coded skeleton overlay video
            try:
                from myogait.video import render_skeleton_video
                overlay_mp4 = str(plots_p / "skeleton_overlay.mp4")
                render_skeleton_video(
                    video_path=str(video_p),
                    data=data,
                    output_path=overlay_mp4,
                    show_angles=True,
                    show_events=True,
                )
                logger.info(f"      Skeleton overlay video saved: {plots_p.relative_to(out_p)}/skeleton_overlay.mp4")
            except Exception as e_vid:
                logger.warning(f"      Could not render skeleton overlay video: {e_vid}")

            logger.info(f"      Plots saved in: {plots_p.relative_to(out_p)}")
        except Exception as e:
            logger.warning(f"      Could not generate plots: {e}")

    # B. PDF Report inside report/
    if generate_pdf:
        pdf_path = str(report_p / "dcm_gait_report.pdf")
        try:
            generate_report(data, cycles_result, stats, pdf_path, language=language)
            logger.info(f"      PDF Report generated: {report_p.relative_to(out_p)}/dcm_gait_report.pdf")
        except Exception as e:
            logger.warning(f"      Could not generate PDF report: {e}")

    # C. Excel Workbook inside data/
    if generate_excel:
        xlsx_path = str(data_p / "dcm_gait_analysis.xlsx")
        try:
            export_excel(data, xlsx_path, cycles_result, stats)
            logger.info(f"      Excel Workbook exported: {data_p.relative_to(out_p)}/dcm_gait_analysis.xlsx")
        except Exception as e:
            logger.warning(f"      Could not export Excel file: {e}")

    # D. CSV Files inside data/csv/
    if generate_csv:
        try:
            csv_files = export_csv(data, str(csv_p), cycles_result, stats)
            logger.info(f"      CSV Tables exported ({len(csv_files)} files in {csv_p.relative_to(out_p)}).")
        except Exception as e:
            logger.warning(f"      Could not export CSV files: {e}")

    # E. OpenSim Kinematics inside opensim/
    if generate_opensim:
        try:
            mot_path = str(opensim_p / "kinematics.mot")
            export_mot(data, mot_path)

            trc_path = str(opensim_p / "markers.trc")
            export_trc(data, trc_path)
            logger.info(f"      OpenSim files exported in: {opensim_p.relative_to(out_p)}")
        except Exception as e:
            logger.warning(f"      Could not export OpenSim files: {e}")

    elapsed = time.time() - t_start

    # F. Generate README.md and run_metadata.json
    _write_run_metadata(
        output_dir=out_p,
        video_path=video_path,
        plane=plane,
        subject_id=subject_id,
        subject_height_m=subject_height_m,
        model=model,
        cutoff_hz=cutoff_hz,
        events_method=events_method,
        stats=stats,
        elapsed_s=elapsed,
    )
    _write_run_readme(
        output_dir=out_p,
        video_path=video_path,
        plane=plane,
        subject_id=subject_id,
        subject_height_m=subject_height_m,
        model=model,
        cutoff_hz=cutoff_hz,
        events_method=events_method,
        stats=stats,
        elapsed_s=elapsed,
    )
    logger.info("      Generated run documentation: README.md & run_metadata.json")

    logger.info("=" * 60)
    logger.info(f"DCM Pipeline completed successfully in {elapsed:.1f}s!")
    logger.info(f"Outputs written to: {out_p.resolve()}")
    logger.info("=" * 60)

    return {
        "data": data,
        "cycles": cycles_result,
        "stats": stats,
        "output_dir": str(out_p.resolve()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run customized DCM Clinical Gait Analysis Pipeline on sagittal or coronal video recordings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=str, help="Path to input video file.")
    group.add_argument("--batch-dir", type=str, help="Directory containing multiple video files to process.")

    parser.add_argument("--output-dir", type=str, default="./Output", help="Base directory to save output reports and files.")
    parser.add_argument("--model", type=str, default="mediapipe", help="Pose estimation model ('mediapipe', 'sapiens2', 'yolo', etc.).")
    parser.add_argument("--cutoff", type=float, default=5.0, help="Low-pass Butterworth filter cutoff in Hz.")
    parser.add_argument("--plane", type=str, choices=["sagittal", "coronal"], default="sagittal", help="Video recording plane.")
    parser.add_argument("--subject-id", type=str, default=None, help="Optional subject ID (e.g. DCM_001).")
    parser.add_argument("--height", type=float, default=None, help="Subject height in meters (e.g. 1.75).")
    parser.add_argument("--events-method", type=str, default="zeni", help="Gait event detection method.")
    parser.add_argument("--language", type=str, choices=["en", "fr"], default="en", help="PDF report language ('en' for English, 'fr' for French).")

    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF report generation.")
    parser.add_argument("--no-excel", action="store_true", help="Skip Excel workbook export.")
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV export.")
    parser.add_argument("--no-opensim", action="store_true", help="Skip OpenSim .mot/.trc export.")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation.")

    args = parser.parse_args()

    if args.video:
        run_dcm_pipeline(
            video_path=args.video,
            output_dir=args.output_dir,
            model=args.model,
            cutoff_hz=args.cutoff,
            plane=args.plane,
            subject_id=args.subject_id,
            subject_height_m=args.height,
            events_method=args.events_method,
            language=args.language,
            generate_pdf=not args.no_pdf,
            generate_excel=not args.no_excel,
            generate_csv=not args.no_csv,
            generate_opensim=not args.no_opensim,
            generate_plots=not args.no_plots,
        )
    elif args.batch_dir:
        batch_p = Path(args.batch_dir)
        extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv")
        video_files = []
        for ext in extensions:
            video_files.extend(list(batch_p.glob(ext)))

        if not video_files:
            logger.error(f"No video files found in {args.batch_dir}")
            sys.exit(1)

        logger.info(f"Found {len(video_files)} video files in {args.batch_dir} for batch processing.")
        for vid in video_files:
            subject_out = Path(args.output_dir) / vid.stem
            try:
                run_dcm_pipeline(
                    video_path=str(vid),
                    output_dir=str(subject_out),
                    model=args.model,
                    cutoff_hz=args.cutoff,
                    plane=args.plane,
                    subject_id=args.subject_id or vid.stem,
                    subject_height_m=args.height,
                    events_method=args.events_method,
                    generate_pdf=not args.no_pdf,
                    generate_excel=not args.no_excel,
                    generate_csv=not args.no_csv,
                    generate_opensim=not args.no_opensim,
                    generate_plots=not args.no_plots,
                )
            except Exception as e:
                logger.error(f"Error processing video {vid.name}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
