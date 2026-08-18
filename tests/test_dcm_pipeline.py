"""Tests for DCM Gait Analysis Pipeline script."""

import sys
from pathlib import Path

# Ensure tests directory and root directory are on sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import make_walking_data  # noqa: E402
import dcm_pipeline  # noqa: E402


def test_dcm_pipeline_imports():
    """Verify dcm_pipeline imports and main entry points exist."""
    assert hasattr(dcm_pipeline, "run_dcm_pipeline")
    assert hasattr(dcm_pipeline, "main")


def test_dcm_pipeline_data_processing(tmp_path):
    """Test DCM processing steps directly on pre-extracted walking data."""
    from myogait import (
        normalize,
        compute_angles,
        compute_extended_angles,
        compute_frontal_angles,
        apply_perspective_correction,
        apply_linear_detrend,
        detect_events,
        segment_cycles,
        analyze_gait,
    )

    data = make_walking_data(n_frames=90, fps=30.0)

    # 1. Normalize
    data = normalize(
        data,
        steps=[
            {"type": "median", "kernel_size": 5},
            {"type": "butterworth", "cutoff": 5.0},
        ],
    )

    # 2. Angles
    data = compute_angles(data)
    data = compute_extended_angles(data)
    data = compute_frontal_angles(data)

    # 3. Corrections (Perspective + Detrend)
    data = apply_perspective_correction(data)
    data = apply_linear_detrend(data)

    assert data["angles"]["perspective_corrected"] is True
    assert data["angles"]["linear_detrended"] is True

    # 4. Events & Cycles
    data = detect_events(data, method="zeni")
    cycles = segment_cycles(data)
    assert "cycles" in cycles

    # 5. Analysis
    stats = analyze_gait(data, cycles)
    assert "spatiotemporal" in stats
    assert "symmetry" in stats


def test_dcm_pipeline_export_generation(tmp_path):
    """Test generating all export files (PDF, Excel, CSV, OpenSim, PNG plots)."""
    from myogait import (
        normalize,
        compute_angles,
        compute_extended_angles,
        compute_frontal_angles,
        apply_perspective_correction,
        apply_linear_detrend,
        detect_events,
        segment_cycles,
        analyze_gait,

    )
    from myogait.scores import (
        gait_variable_scores,
        gait_profile_score_2d,
        sagittal_deviation_index,
    )
    from myogait.export import export_csv, export_excel, export_mot, export_trc
    from myogait.report import generate_report

    data = make_walking_data(n_frames=90, fps=30.0)
    data = normalize(data, steps=[{"type": "median", "kernel_size": 5}, {"type": "butterworth", "cutoff": 5.0}])
    data = compute_angles(data)
    data = compute_extended_angles(data)
    data = compute_frontal_angles(data)
    data = apply_perspective_correction(data)
    data = apply_linear_detrend(data)
    data = detect_events(data, method="zeni")
    cycles_result = segment_cycles(data)
    stats = analyze_gait(data, cycles_result)

    stats["gvs"] = gait_variable_scores(cycles_result)
    stats["gps"] = gait_profile_score_2d(cycles_result)
    stats["sdi"] = sagittal_deviation_index(cycles_result)

    # Test PDF report
    pdf_path = str(tmp_path / "report.pdf")
    generate_report(data, cycles_result, stats, pdf_path)
    assert Path(pdf_path).exists() and Path(pdf_path).stat().st_size > 0

    # Test Excel export
    xlsx_path = str(tmp_path / "gait.xlsx")
    export_excel(data, xlsx_path, cycles_result, stats)
    assert Path(xlsx_path).exists() and Path(xlsx_path).stat().st_size > 0

    # Test CSV export
    csv_files = export_csv(data, str(tmp_path), cycles_result, stats)
    assert len(csv_files) > 0

    # Test OpenSim exports
    mot_path = str(tmp_path / "kinematics.mot")
    export_mot(data, mot_path)
    assert Path(mot_path).exists() and Path(mot_path).stat().st_size > 0

    trc_path = str(tmp_path / "markers.trc")
    export_trc(data, trc_path)
    assert Path(trc_path).exists() and Path(trc_path).stat().st_size > 0


def test_dcm_pipeline_folder_structure_and_metadata(tmp_path):
    """Test that README.md and run_metadata.json are generated cleanly."""
    from dcm_pipeline import _write_run_readme, _write_run_metadata

    _data = make_walking_data(n_frames=90, fps=30.0)  # noqa: F841
    stats = {
        "spatiotemporal": {
            "cadence_steps_per_min": 100.0,
            "stride_time_mean_s": 1.2,
            "stride_time_std_s": 0.05,
            "stance_pct_left": 60.0,
            "stance_pct_right": 61.0,
        },
        "symmetry": {"overall_si": 5.2},
        "gps": {"gps_overall": 6.8},
        "sdi": 94.0,
        "pathology_flags": ["Reduced knee flexion peak"],
    }

    _write_run_metadata(
        output_dir=tmp_path,
        video_path="patient01.mp4",
        plane="sagittal",
        subject_id="DCM_001",
        subject_height_m=1.75,
        model="mediapipe",
        cutoff_hz=5.0,
        events_method="zeni",
        stats=stats,
        elapsed_s=12.5,
        apply_bias_correction=False,
    )
    _write_run_readme(
        output_dir=tmp_path,
        video_path="patient01.mp4",
        plane="sagittal",
        subject_id="DCM_001",
        subject_height_m=1.75,
        model="mediapipe",
        cutoff_hz=5.0,
        events_method="zeni",
        stats=stats,
        elapsed_s=12.5,
        apply_bias_correction=False,
    )

    readme = tmp_path / "README.md"
    meta = tmp_path / "run_metadata.json"

    readme_text = readme.read_text(encoding="utf-8")
    assert readme.exists() and "DCM Gait Analysis Run Report" in readme_text
    assert "**Cycle Analysis**: Both sides segmented normally." in readme_text

    meta_text = meta.read_text(encoding="utf-8")
    assert meta.exists() and "myogait_version" in meta_text
    import json
    meta_json = json.loads(meta_text)
    assert meta_json.get("skipped_sides") == []


def test_dcm_pipeline_folder_structure_and_metadata_skipped_sides(tmp_path):
    """Test that README.md and run_metadata.json handle skipped sides correctly."""
    from dcm_pipeline import _write_run_readme, _write_run_metadata
    import json

    _data = make_walking_data(n_frames=90, fps=30.0)  # noqa: F841
    stats = {
        "spatiotemporal": {
            "cadence_steps_per_min": 100.0,
            "stride_time_mean_s": 1.2,
            "stride_time_std_s": 0.05,
            "stance_pct_left": 60.0,
            "stance_pct_right": 60.0,
        },
        "symmetry": {"overall_si": 2.5},
        "gps": {"gps_2d_overall": 5.5},
        "sdi": {"gdi_2d_overall": 95.0},
        "pathology_flags": [],
    }

    skipped_sides = [{"side": "right", "reason": "insufficient_hs_events", "n_hs": 1}]

    _write_run_metadata(
        output_dir=tmp_path,
        video_path="patient01.mp4",
        plane="sagittal",
        subject_id="DCM_001",
        subject_height_m=1.75,
        model="mediapipe",
        cutoff_hz=5.0,
        events_method="zeni",
        stats=stats,
        elapsed_s=12.5,
        apply_bias_correction=False,
        skipped_sides=skipped_sides,
    )
    _write_run_readme(
        output_dir=tmp_path,
        video_path="patient01.mp4",
        plane="sagittal",
        subject_id="DCM_001",
        subject_height_m=1.75,
        model="mediapipe",
        cutoff_hz=5.0,
        events_method="zeni",
        stats=stats,
        elapsed_s=12.5,
        apply_bias_correction=False,
        skipped_sides=skipped_sides,
    )

    readme = tmp_path / "README.md"
    meta = tmp_path / "run_metadata.json"

    readme_text = readme.read_text(encoding="utf-8")
    assert "Cycle Analysis Warning" in readme_text
    assert "Right side skipped" in readme_text

    meta_json = json.loads(meta.read_text(encoding="utf-8"))
    assert meta_json.get("skipped_sides") == skipped_sides


