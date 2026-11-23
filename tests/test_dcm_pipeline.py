"""Tests for the DCM clinical pipeline (dcm_pipeline.py).

The central guarantee: with default settings every number equals
``myogait.run_pipeline``; side labels follow the patient's true sides
without changing any per-leg value.
"""

import copy
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import myogait  # noqa: E402
import dcm_pipeline as dp  # noqa: E402
from conftest import make_walking_data  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────────


def _canon(x):
    """JSON-canonical form for exact comparison (NaN-safe)."""
    return json.dumps(x, sort_keys=True, default=str)


def _without_scores(stats):
    return {k: v for k, v in stats.items() if k not in ("gvs", "gps", "sdi")}


def _mirror(data, swap_labels, anatomical):
    """Emulate extract()'s mirroring of a left-to-right walk."""
    d = copy.deepcopy(data)
    ext = d.setdefault("extraction", {})
    ext["was_flipped"] = True
    if anatomical:
        ext["side_label_convention"] = "anatomical"
    for f in d["frames"]:
        lm = f["landmarks"]
        for v in lm.values():
            v["x"] = 1.0 - v["x"]
        if swap_labels:
            f["landmarks"] = dp._swap_side_names(lm)
    return d


def _oblique(data):
    """Hips/shoulders visibly apart with a pelvic drop, so pelvic tilt is defined."""
    d = copy.deepcopy(data)
    for i, f in enumerate(d["frames"]):
        lm = f["landmarks"]
        for side, dx in (("LEFT", -0.12), ("RIGHT", 0.12)):
            for j in ("HIP", "SHOULDER"):
                lm[f"{side}_{j}"]["x"] += dx
        lm["RIGHT_HIP"]["y"] -= 0.02 * np.sin(i / 10)
    return d


def _series(data, key):
    return np.array([np.nan if af.get(key) is None else af[key]
                     for af in data["angles"]["frames"]], float)


OTHER = {"left": "right", "right": "left", "L": "R", "R": "L"}


@pytest.fixture(scope="module")
def walk():
    return make_walking_data(n_frames=600, fps=30.0)


# ── settings registry ────────────────────────────────────────────────────


def test_entry_points_exist():
    assert callable(dp.run_dcm_pipeline)
    assert callable(dp.main)


def test_registry_defaults_equal_myogait_defaults():
    """Every myogait-backed default is read from myogait's own signatures."""
    rp = inspect.signature(myogait.run_pipeline).parameters
    for key in ("model", "butterworth_cutoff", "event_method", "min_cycle_duration_s",
                "max_cycle_duration_s", "n_points", "direction_filter", "outlier_rejection"):
        assert dp.SETTINGS[key].default == rp[key].default, key
    ex = inspect.signature(myogait.extract).parameters
    for key in ("flip_if_right", "correct_inversions", "edge_margin", "min_leg_vis", "adaptive_leg_vis"):
        assert dp.SETTINGS[key].default == ex[key].default, key
    from myogait.normative import STRATA
    gvs = inspect.signature(myogait.scores.gait_variable_scores).parameters
    assert dp.SETTINGS["normative_stratum"].default == gvs["stratum"].default
    assert set(dp.SETTINGS["normative_stratum"].choices) == set(STRATA)
    for key in ("median_spike_filter", "perspective_correction", "linear_detrend",
                "bias_correction", "adaptive_leg_vis"):
        assert dp.SETTINGS[key].default is False, key
    assert dp.SETTINGS["visible_side"].default == "both"


def test_every_setting_is_documented():
    for spec in dp.SETTINGS_LIST:
        assert spec.group in dp.GROUPS
        assert len(spec.help) > 40, spec.key
        assert spec.label


def test_standard_frame_preset_is_myogait_default():
    standard = dp.FRAME_QUALITY_PRESETS["Standard (myogait default)"]
    for key, value in standard.items():
        assert dp.SETTINGS[key].default == value


def test_resolve_settings_aliases_and_validation():
    s = dp.resolve_settings({"cutoff_hz": 6, "events_method": "velocity", "language": "fr"})
    assert s["butterworth_cutoff"] == 6.0 and s["event_method"] == "velocity"
    assert s["report_language"] == "fr"
    with pytest.raises(ValueError):
        dp.resolve_settings({"not_a_setting": 1})
    with pytest.raises(ValueError):
        dp.resolve_settings({"visible_side": "middle"})
    with pytest.raises(ValueError):
        dp.resolve_settings({"min_cycle_duration_s": 2.0, "max_cycle_duration_s": 1.5})


def test_deviations_listing():
    assert dp.calculation_deviations(dp.default_settings()) == []
    s = dp.resolve_settings({"max_cycle_duration_s": 2.5, "report_language": "fr"})
    assert [d["setting"] for d in dp.calculation_deviations(s)] == ["max_cycle_duration_s"]
    assert {d["setting"] for d in dp.deviations_from_defaults(s)} == {"max_cycle_duration_s", "report_language"}


# ── equivalence with myogait.run_pipeline ────────────────────────────────


def test_defaults_are_identical_to_run_pipeline(walk):
    ours = dp.compute_dcm_gait(copy.deepcopy(walk), dp.default_settings())
    ref = myogait.run_pipeline(copy.deepcopy(walk))
    assert ours["chain"] == "myogait.run_pipeline"
    assert _canon(_without_scores(ours["stats"])) == _canon(ref["stats"])
    assert _canon(ours["cycles"]) == _canon(ref["cycles"])
    assert _canon(ours["quality"]) == _canon(ref["quality"])
    core = ("hip_L", "hip_R", "knee_L", "knee_R", "ankle_L", "ankle_R", "trunk_angle", "pelvis_tilt")
    for key in core:
        np.testing.assert_array_equal(_series(ours["data"], key), _series(ref["data"], key))


def test_manual_chain_is_identical_to_run_pipeline(walk):
    """The step-by-step chain used when a correction is on must not drift."""
    ours = dp.compute_dcm_gait(copy.deepcopy(walk), dp.default_settings(), force_manual=True)
    ref = myogait.run_pipeline(copy.deepcopy(walk))
    assert _canon(_without_scores(ours["stats"])) == _canon(ref["stats"])
    assert _canon(ours["cycles"]) == _canon(ref["cycles"])


def test_scores_use_myogait_functions(walk):
    ours = dp.compute_dcm_gait(copy.deepcopy(walk), dp.default_settings())
    ref = myogait.run_pipeline(copy.deepcopy(walk))
    assert _canon(ours["stats"]["gps"]) == _canon(myogait.scores.gait_profile_score_2d(ref["cycles"]))
    assert _canon(ours["stats"]["sdi"]) == _canon(myogait.scores.sagittal_deviation_index(ref["cycles"]))


def test_optional_correction_changes_numbers_and_is_reported(walk):
    s = dp.resolve_settings({"linear_detrend": True})
    ours = dp.compute_dcm_gait(copy.deepcopy(walk), s)
    assert "linear_detrend" in ours["chain"]
    assert ours["data"]["angles"].get("linear_detrended") is True


# ── side labels ──────────────────────────────────────────────────────────


def test_anatomical_labels_keep_every_per_leg_number(walk):
    """Same numbers as myogait for each physical leg, filed under the true side."""
    ours = dp.compute_dcm_gait(_mirror(walk, swap_labels=False, anatomical=True), dp.default_settings())
    ref = myogait.run_pipeline(_mirror(walk, swap_labels=True, anatomical=False))
    for joint in ("hip", "knee", "ankle"):
        for side in ("L", "R"):
            np.testing.assert_array_equal(_series(ours["data"], f"{joint}_{side}"),
                                          _series(ref["data"], f"{joint}_{OTHER[side]}"))

    def spans(cycles, side):
        return [(c["start_frame"], c["toe_off_frame"], c["end_frame"])
                for c in cycles["cycles"] if c["side"] == side]
    for side in ("left", "right"):
        assert spans(ours["cycles"], side) == spans(ref["cycles"], OTHER[side])
    st_o, st_r = ours["stats"]["spatiotemporal"], ref["stats"]["spatiotemporal"]
    for key, value in st_r.items():
        mapped = key.replace("left", "@").replace("right", "left").replace("@", "right")
        assert st_o[mapped] == value, key


def test_pelvis_sign_follows_true_side_on_mirrored_clip(walk):
    oblique = _oblique(walk)
    ours = dp.compute_dcm_gait(_mirror(oblique, swap_labels=False, anatomical=True), dp.default_settings())
    ref = myogait.run_pipeline(_mirror(oblique, swap_labels=True, anatomical=False))
    original = myogait.compute_angles(copy.deepcopy(myogait.normalize(
        copy.deepcopy(oblique), filters=["butterworth"], butterworth_cutoff=8.0)), calibrate=False)
    assert ours["pelvis_relabelled"] is True
    tilt = _series(ours["data"], "pelvis_tilt")
    assert np.isfinite(tilt).all() and np.abs(tilt).max() < 10
    np.testing.assert_array_equal(tilt, -_series(ref["data"], "pelvis_tilt"))
    np.testing.assert_allclose(tilt, _series(original, "pelvis_tilt"), atol=1e-9)
    np.testing.assert_array_equal(_series(ours["data"], "pelvis_obliquity"), tilt)
    trend_o = [p for p in ours["stats"]["pathologies"] if p["pattern"] == "trendelenburg"]
    trend_r = [p for p in ref["stats"]["pathologies"] if p["pattern"] == "trendelenburg"]
    assert _canon(trend_o) == _canon(trend_r)


def test_pelvis_untouched_when_not_mirrored(walk):
    oblique = _oblique(walk)
    ours = dp.compute_dcm_gait(copy.deepcopy(oblique), dp.default_settings())
    ref = myogait.run_pipeline(copy.deepcopy(oblique))
    assert ours["pelvis_relabelled"] is False
    np.testing.assert_array_equal(_series(ours["data"], "pelvis_tilt"), _series(ref["data"], "pelvis_tilt"))


def test_flip_keeps_anatomical_labels():
    from myogait.constants import MP_NAME_TO_INDEX
    from myogait.extract import SIDE_LABEL_CONVENTION, _flip_landmarks
    assert SIDE_LABEL_CONVENTION == "anatomical"
    lm = np.full((33, 3), 0.5)
    lm[MP_NAME_TO_INDEX["LEFT_KNEE"]] = [0.2, 0.6, 0.9]
    out = _flip_landmarks(lm)
    np.testing.assert_allclose(out[MP_NAME_TO_INDEX["LEFT_KNEE"]], [0.8, 0.6, 0.9])


# ── side selection, slow walkers ─────────────────────────────────────────


def test_visible_side_uses_myogait_on_that_side_only(walk):
    s = dp.resolve_settings({"visible_side": "left"})
    ours = dp.compute_dcm_gait(copy.deepcopy(walk), s)
    ref = myogait.run_pipeline(copy.deepcopy(walk))
    left = [c for c in ref["cycles"]["cycles"] if c["side"] == "left"]
    expected = myogait.analyze_gait(ref["data"], {"cycles": left, "summary": myogait.summarize_cycles(left)})
    assert _canon(_without_scores(ours["stats"])) == _canon(expected)
    assert len(ours["cycles"]["cycles"]) == len(ref["cycles"]["cycles"])  # both sides kept in data
    view, view_cycles = dp._mask_for_display(ours["data"], ours["cycles"], "left")
    assert all("hip_R" not in af for af in view["angles"]["frames"])
    assert {c["side"] for c in view_cycles["cycles"]} == {"left"}


def test_slow_walker_stride_window():
    slow = make_walking_data(n_frames=600, fps=30.0)
    slow["meta"]["fps"] = 15.0  # same motion, half speed: ~2 s strides
    default = dp.compute_dcm_gait(copy.deepcopy(slow), dp.default_settings())
    assert default["cycles"]["cycles"] == []
    assert {s["reason"] for s in default["skipped_sides"]} == {"no_valid_cycles"}
    widened = dp.compute_dcm_gait(copy.deepcopy(slow), dp.resolve_settings({"max_cycle_duration_s": 2.5}))
    assert len(widened["cycles"]["cycles"]) > 0


# ── full run, documentation, command line ────────────────────────────────


@pytest.fixture()
def pose_json(tmp_path, walk):
    path = tmp_path / "walk.json"
    path.write_text(json.dumps(walk), encoding="utf-8")
    return path


def test_full_run_outputs_and_documentation(tmp_path, pose_json):
    res = dp.run_dcm_pipeline(input_json_path=str(pose_json), output_dir=str(tmp_path / "out"),
                              subject_id="DCM_001", subject_height_m=1.72, plane="coronal",
                              generate_videos=False)
    out = Path(res["output_dir"])
    assert out.name.startswith("DCM_001_coronal_sapiens2-quick_")
    for rel in ("README.md", "run_metadata.json", "report/dcm_gait_report.pdf",
                "data/dcm_gait_data.json", "data/dcm_gait_analysis.xlsx",
                "opensim/kinematics.mot", "opensim/markers.trc", "plots/summary.png"):
        assert (out / rel).exists(), rel
    assert res["outputs_failed"] == []
    meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["myogait_version"] == myogait.__version__
    assert meta["calculation_chain"] == "myogait.run_pipeline"
    assert meta["deviations_from_myogait_defaults"] == []
    assert meta["settings"]["plane"] == "coronal"
    assert meta["results_summary"]["gait_profile_score_gps"] is not None
    saved = json.loads((out / "data/dcm_gait_data.json").read_text(encoding="utf-8"))
    assert saved["meta"]["plane"] == "coronal"
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "All calculation settings are myogait defaults" in readme


def test_full_run_documents_deviations(tmp_path, pose_json):
    res = dp.run_dcm_pipeline(input_json_path=str(pose_json), output_dir=str(tmp_path),
                              max_cycle_duration_s=2.5, generate_videos=False, generate_pdf=False)
    out = Path(res["output_dir"])
    meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    assert [d["setting"] for d in meta["deviations_from_myogait_defaults"]] == ["max_cycle_duration_s"]
    assert "Settings changed from myogait defaults" in (out / "README.md").read_text(encoding="utf-8")


def test_side_label_text():
    right = dp._side_label_text({"side_label_convention": "anatomical", "direction_detected": "right"})
    assert "patient's **right** leg was nearest the camera" in right
    left = dp._side_label_text({"side_label_convention": "anatomical", "direction_detected": "left"})
    assert "patient's **left** leg was nearest the camera" in left
    assert "unknown" in dp._side_label_text({"side_label_convention": "anatomical"})
    assert "standard myogait" in dp._side_label_text({})


def test_cli_json_with_overrides(tmp_path, pose_json):
    dp.main(["--json", str(pose_json), "--output-dir", str(tmp_path), "--cutoff", "6",
             "--max-cycle-duration-s", "2.5", "--no-outlier-rejection", "--language", "fr",
             "--no-pdf", "--no-videos", "--no-opensim"])
    (run_dir,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    s = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))["settings"]
    assert s["butterworth_cutoff"] == 6.0 and s["max_cycle_duration_s"] == 2.5
    assert s["outlier_rejection"] is False and s["report_language"] == "fr"


def test_cli_video_mode_no_longer_crashes_on_arguments(tmp_path):
    """--video used to raise TypeError (unexpected 'language' argument)."""
    fake = tmp_path / "fake.mp4"
    fake.write_bytes(b"not a video")
    with pytest.raises(Exception) as err:
        dp.main(["--video", str(fake), "--output-dir", str(tmp_path), "--model", "mediapipe"])
    assert not isinstance(err.value, TypeError)


def test_cli_list_settings(capsys):
    dp.main(["--list-settings"])
    out = capsys.readouterr().out
    for key in dp.SETTINGS:
        assert "--" + key.replace("_", "-") in out


def test_dashboard_builds_from_registry():
    import ast
    src = (root_dir / "dcm_dashboard.py").read_text(encoding="utf-8")
    ast.parse(src)
    assert "SETTINGS_LIST" in src and "FRAME_QUALITY_PRESETS" in src
    assert "FRAME_QUALITY_PRESETS = {" not in src  # single source of truth: dcm_pipeline
