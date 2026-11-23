#!/usr/bin/env python3
"""DCM Gait Analysis Pipeline.

A clinical wrapper around myogait for Degenerative Cervical Myelopathy (DCM)
and pathological gait, designed for 5 m walkway recordings (sagittal or
coronal views).

Calculations
------------
Every number is computed by myogait itself. With the default settings, the
processing chain is exactly :func:`myogait.run_pipeline` (the recipe that
myogait validated against optical motion capture): 8 Hz Butterworth filter,
sagittal angles without neutral calibration, flexion-positive sign
convention, Zeni gait events, 0.8-1.6 s cycles, direction-consistent cycle
filter, outlier-cycle rejection and ``analyze_gait``. The test suite checks
that the outputs are identical to ``run_pipeline``.

Every setting is listed in :data:`SETTINGS` with its myogait default and a
plain-language explanation. The dashboard, the command line and
``run_metadata.json`` all read this list; any setting changed from the
myogait default is recorded as a deviation in the run documentation.

Side labels
-----------
Left/right always refer to the patient's true anatomical sides (myogait
itself swaps the labels when it mirrors a left-to-right walk). Per-leg numbers
are identical to myogait's, only filed under the correct side; pelvic
tilt/obliquity keeps myogait's magnitude with the sign of the true side
("positive = patient's right side up"). See :func:`_restore_anatomical_pelvis`.

Outputs
-------
PDF report, Excel workbook, CSV tables, OpenSim (.mot/.trc), PNG plots,
skeleton-overlay and anonymised stick-figure videos, plus an auto-generated
README.md and run_metadata.json for auditing.

Usage
-----
    python dcm_pipeline.py --video walk.mp4 --subject-id DCM_001 --height 1.72
    python dcm_pipeline.py --batch-dir ./patient_videos --output-dir ./batch_results
    python dcm_pipeline.py --video walk.mp4 --max-cycle-duration-s 2.5   # slow walker
    python dcm_pipeline.py --list-settings
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for plots
import matplotlib.pyplot as plt  # noqa: E402

import myogait  # noqa: E402
from myogait import (  # noqa: E402
    analyze_gait,
    apply_linear_detrend,
    apply_perspective_correction,
    canonicalize_angle_signs,
    compute_angles,
    compute_extended_angles,
    compute_frontal_angles,
    detect_events,
    extract,
    load_json,
    normalize,
    reject_outlier_cycles,
    run_pipeline,
    save_json,
    segment_cycles,
    set_subject,
    summarize_cycles,
)
from myogait.corrections import (  # noqa: E402
    apply_hip_bias_correction,
    apply_knee_bias_correction,
)
from myogait.export import export_csv, export_excel, export_mot, export_trc  # noqa: E402
from myogait.pipeline import (  # noqa: E402
    _diagnose,
    enforce_flexion_positive,
    filter_cycles_by_direction,
)
from myogait.plotting import plot_angles, plot_cycles, plot_events, plot_summary  # noqa: E402
from myogait.report import generate_report  # noqa: E402
from myogait.scores import (  # noqa: E402
    gait_profile_score_2d,
    gait_variable_scores,
    sagittal_deviation_index,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dcm_pipeline")


# =============================================================================
# Settings registry: one place for every adjustable parameter
# =============================================================================


@dataclass(frozen=True)
class Setting:
    """One adjustable pipeline parameter.

    ``default`` is myogait's own default (``source`` says where it comes
    from). Settings whose ``source`` is ``"DCM"`` are not myogait parameters
    (labels, outputs, optional extras); they are still listed so that every
    choice is visible and documented.
    """

    key: str
    default: Any
    kind: str  # "float" | "int" | "bool" | "choice" | "model"
    group: str
    label: str
    help: str
    source: str
    choices: Tuple[Any, ...] = ()
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    changes_numbers: bool = True
    extras: Dict[str, Any] = field(default_factory=dict)


GROUPS = (
    "Pose extraction",
    "Frame quality",
    "Signal filtering",
    "Gait events & cycles",
    "Side selection",
    "Clinical scores",
    "Optional corrections (not in myogait's validated recipe)",
    "Report",
)

SETTINGS_LIST: List[Setting] = [
    # ── Pose extraction ─────────────────────────────────────────────────
    Setting(
        "model", "sapiens2-quick", "model", "Pose extraction", "Pose model",
        "The AI model that finds the joints in each video frame. 'sapiens2-quick' is "
        "myogait's default and the model it validated against motion capture; it needs an "
        "NVIDIA GPU. 'mediapipe' runs on CPU (fast, less accurate). Larger Sapiens 2 models "
        "(mid/top/ultra) are more accurate but need more GPU memory. Compare patients and "
        "sessions only when they were processed with the same model.",
        "myogait.run_pipeline(model=...)",
    ),
    Setting(
        "flip_if_right", True, "bool", "Pose extraction", "Normalise walking direction",
        "When the patient walks left-to-right, mirror the coordinates so every recording is "
        "analysed in the same direction (myogait default). Labels still follow the patient's "
        "true sides. Leave this on for side-view (sagittal) videos. For coronal videos (walking "
        "towards/away from the camera) the direction is ambiguous; turning it off avoids an "
        "accidental mirror.",
        "myogait.extract(flip_if_right=...)",
    ),
    Setting(
        "correct_inversions", True, "bool", "Pose extraction", "Fix left/right tracking swaps",
        "Pose models sometimes swap the two legs for a few frames in a side view (the far "
        "leg is hidden). This detects those swaps from the motion and corrects them "
        "(myogait default). Turn off only to inspect the raw model output.",
        "myogait.extract(correct_inversions=...)",
    ),
    # ── Frame quality ───────────────────────────────────────────────────
    Setting(
        "edge_margin", 0.02, "float", "Frame quality", "Edge margin (fraction of frame)",
        "Frames at the start/end of the clip are discarded while a hip, knee or ankle is "
        "within this distance of the image border (0.02 = 2 %, myogait default). Lower it "
        "only if the patient walks very close to the edge of the frame.",
        "myogait.extract (hard-coded 0.02)", min=0.0, max=0.2, step=0.005,
    ),
    Setting(
        "min_leg_vis", 0.30, "float", "Frame quality", "Minimum leg visibility",
        "Frames at the start/end of the clip are discarded while the model's confidence in "
        "any hip, knee or ankle is below this value (0.30, myogait default). Lower it for a "
        "distant or poorly lit patient whose frames are being dropped; the trade-off is less "
        "reliable gait events at the clip ends.",
        "myogait.extract (hard-coded 0.3)", min=0.0, max=1.0, step=0.01,
    ),
    Setting(
        "adaptive_leg_vis", False, "bool", "Frame quality", "Adaptive visibility floor",
        "Scale the visibility threshold to this clip's own typical leg confidence: "
        "max(minimum visibility, 0.5 x median leg confidence). Off in myogait. Useful together "
        "with a low minimum visibility for distant subjects.",
        "DCM extension of myogait.extract (off = myogait behaviour)",
    ),
    # ── Signal filtering ────────────────────────────────────────────────
    Setting(
        "butterworth_cutoff", 8.0, "float", "Signal filtering", "Low-pass filter cutoff (Hz)",
        "Smooths the joint trajectories (zero-phase Butterworth, order 2). myogait's default "
        "8 Hz keeps the fast ankle push-off and knee swing peaks; myogait measured that 4 Hz "
        "flattened them by 7-10°. Lower values give smoother but flatter curves.",
        "myogait.run_pipeline(butterworth_cutoff=...)", min=1.0, max=15.0, step=0.5,
    ),
    Setting(
        "median_spike_filter", False, "bool", "Signal filtering", "Median spike filter",
        "Apply a median filter before the low-pass filter to remove single-frame tracking "
        "spikes. Not part of myogait's validated recipe (off by default). Useful for noisy "
        "videos with obvious jumps of the skeleton.",
        "DCM option (myogait.normalize 'median' step)",
    ),
    Setting(
        "median_kernel", 5, "int", "Signal filtering", "Median filter window (frames)",
        "Window of the median spike filter, in frames (odd number). Only used when the median "
        "spike filter is on.",
        "DCM option", min=3, max=15, step=2,
    ),
    # ── Gait events & cycles ────────────────────────────────────────────
    Setting(
        "event_method", "zeni", "choice", "Gait events & cycles", "Gait event method",
        "Algorithm that finds heel strikes and toe-offs. 'zeni' (foot position relative to "
        "the pelvis) is myogait's validated default. Others: crossing, velocity, oconnor and "
        "the gaitkit 'gk_' detectors.",
        "myogait.run_pipeline(event_method=...)",
        extras={"choices_from": "myogait.events.EVENT_METHODS"},
    ),
    Setting(
        "min_cycle_duration_s", 0.8, "float", "Gait events & cycles", "Shortest stride kept (s)",
        "Strides (heel strike to next heel strike of the same foot) shorter than this are "
        "discarded as detection errors. myogait default 0.8 s.",
        "myogait.run_pipeline(min_cycle_duration_s=...)", min=0.2, max=2.0, step=0.05,
    ),
    Setting(
        "max_cycle_duration_s", 1.6, "float", "Gait events & cycles", "Longest stride kept (s)",
        "Strides longer than this are discarded. myogait default 1.6 s (about 75 steps/min). "
        "Severe DCM patients can walk slower than this: if the run reports dropped or missing "
        "cycles for a slow walker, raise it (e.g. 2.5 s, myogait's own segment_cycles default).",
        "myogait.run_pipeline(max_cycle_duration_s=...)", min=0.8, max=5.0, step=0.1,
    ),
    Setting(
        "n_points", 101, "int", "Gait events & cycles", "Points per normalised cycle",
        "Each stride is resampled to this many points (0-100 % of the gait cycle). myogait "
        "default 101 (one point per percent). Changing it is rarely useful.",
        "myogait.run_pipeline(n_points=...)", min=51, max=201, step=10,
    ),
    Setting(
        "direction_filter", True, "bool", "Gait events & cycles", "Keep one walking direction",
        "On there-and-back walkways, keep only the cycles walked in the dominant direction "
        "(myogait default). Turn off only if the patient walks one way and cycles are "
        "unexpectedly dropped.",
        "myogait.run_pipeline(direction_filter=...)",
    ),
    Setting(
        "outlier_rejection", True, "bool", "Gait events & cycles", "Reject atypical cycles",
        "Drop cycles whose hip, knee or ankle curve is far from the patient's own median "
        "curve, typically strides taken while entering or leaving the frame (myogait "
        "default; only applies when a side has 5 or more cycles).",
        "myogait.run_pipeline(outlier_rejection=...)",
    ),
    # ── Side selection ──────────────────────────────────────────────────
    Setting(
        "visible_side", "both", "choice", "Side selection", "Side to analyse",
        "'both' (myogait default) analyses both legs. In a side view the far leg is partly "
        "hidden; choosing the leg nearest the camera computes the statistics from that leg's "
        "cycles only and hides the other leg in plots, videos and the report. Symmetry and "
        "double support need both legs and are then unavailable. The run README tells you "
        "which leg faced the camera.",
        "DCM option (myogait computes both sides)", choices=("both", "left", "right"),
    ),
    # ── Clinical scores ─────────────────────────────────────────────────
    Setting(
        "normative_stratum", "adult", "choice", "Clinical scores", "Normative reference group",
        "Reference curves used for the Gait Profile Score, Gait Variable Scores and Sagittal "
        "Deviation Index. myogait default 'adult'; 'elderly' scales the adult norms for older "
        "subjects.",
        "myogait.scores (stratum='adult')", choices=("adult", "elderly", "pediatric"),
    ),
    # ── Optional corrections ────────────────────────────────────────────
    Setting(
        "perspective_correction", False, "bool",
        "Optional corrections (not in myogait's validated recipe)", "Perspective correction",
        "Corrects hip and knee angles for foreshortening when the leg is not parallel to the "
        "camera (myogait's apply_perspective_correction). Not part of the validated recipe; "
        "myogait suggests it in the run warnings when it detects out-of-plane distortion.",
        "DCM option (myogait.apply_perspective_correction)",
    ),
    Setting(
        "linear_detrend", False, "bool",
        "Optional corrections (not in myogait's validated recipe)", "Linear drift removal",
        "Removes a slow linear drift of the joint angles over the clip, caused by the "
        "distance to the camera changing along the walkway (myogait's apply_linear_detrend). "
        "Not part of the validated recipe.",
        "DCM option (myogait.apply_linear_detrend)",
    ),
    Setting(
        "bias_correction", False, "bool",
        "Optional corrections (not in myogait's validated recipe)", "Healthy-adult bias correction",
        "Applies myogait's hip and knee bias corrections learnt on HEALTHY adults. myogait "
        "warns not to use them for clinical reading of pathological gait: they inject a "
        "healthy pattern and can mask compensations. Keep off for DCM patients.",
        "DCM option (myogait.corrections.apply_hip/knee_bias_correction)",
    ),
    # ── Report ──────────────────────────────────────────────────────────
    Setting(
        "plane", "sagittal", "choice", "Report", "Recording plane",
        "How the video was filmed: 'sagittal' (side view) or 'coronal' (front/back view). "
        "Recorded in the outputs; it does not change any calculation.",
        "DCM metadata", choices=("sagittal", "coronal"), changes_numbers=False,
    ),
    Setting(
        "report_language", "en", "choice", "Report", "PDF report language",
        "Language of the PDF report (English or French). Only the text changes, not the numbers.",
        "DCM option", choices=("en", "fr"),
        changes_numbers=False,
    ),
    Setting(
        "extended_angles", True, "bool", "Report", "Extra angles (head, arms, pelvis)",
        "Also compute myogait's extended angles (head posture, shoulder and elbow flexion, "
        "sagittal pelvic tilt) for the exports. They are computed after the statistics and do "
        "not change any other number.",
        "DCM option (myogait.compute_extended_angles)", changes_numbers=False,
    ),
]

SETTINGS: Dict[str, Setting] = {s.key: s for s in SETTINGS_LIST}

#: Old keyword names accepted for backward compatibility.
SETTING_ALIASES = {
    "cutoff_hz": "butterworth_cutoff",
    "events_method": "event_method",
    "apply_bias_correction": "bias_correction",
    "language": "report_language",
}

#: One-click values for the three frame-quality settings. "Standard" is
#: myogait's own behaviour.
FRAME_QUALITY_PRESETS = {
    "Standard (myogait default)": {"edge_margin": 0.02, "min_leg_vis": 0.30, "adaptive_leg_vis": False},
    "Distant subject": {"edge_margin": 0.005, "min_leg_vis": 0.05, "adaptive_leg_vis": True},
}

#: Settings that switch the pipeline away from the plain run_pipeline call.
_EXTRA_STEP_KEYS = ("median_spike_filter", "perspective_correction",
                    "linear_detrend", "bias_correction")


def default_settings() -> Dict[str, Any]:
    """Return a fresh dict of every setting at its default value."""
    return {k: s.default for k, s in SETTINGS.items()}


def resolve_settings(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Merge *overrides* (aliases allowed) into the defaults and validate."""
    settings = default_settings()
    for key, value in (overrides or {}).items():
        key = SETTING_ALIASES.get(key, key)
        if key == "enable_bilateral":  # legacy flag, now derived from visible_side
            continue
        if key not in SETTINGS:
            raise ValueError(f"Unknown setting '{key}'. Run with --list-settings to see them.")
        if value is None:
            continue
        spec = SETTINGS[key]
        if spec.kind == "float":
            value = float(value)
        elif spec.kind == "int":
            value = int(value)
        elif spec.kind == "bool":
            value = bool(value)
        if spec.choices and value not in spec.choices:
            raise ValueError(f"Setting '{key}' must be one of {spec.choices}, got {value!r}.")
        settings[key] = value
    if settings["min_cycle_duration_s"] >= settings["max_cycle_duration_s"]:
        raise ValueError("Shortest stride kept must be smaller than the longest stride kept.")
    return settings


def deviations_from_defaults(settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    """List the settings that differ from their default, for documentation."""
    out = []
    for key, spec in SETTINGS.items():
        if settings.get(key) != spec.default:
            out.append({
                "setting": key,
                "label": spec.label,
                "value": settings.get(key),
                "default": spec.default,
                "changes_numbers": spec.changes_numbers,
            })
    return out


def calculation_deviations(settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Deviations that change computed numbers (vs plain myogait defaults)."""
    return [d for d in deviations_from_defaults(settings) if d["changes_numbers"]]


# =============================================================================
# Calculation chain
# =============================================================================


def _run_myogait_chain(data: dict, s: Dict[str, Any], force_manual: bool = False):
    """Run myogait's validated recipe, with the optional DCM corrections.

    With no optional correction enabled this is literally
    ``myogait.run_pipeline``. When a correction is enabled the same steps as
    ``myogait.pipeline._run_steps`` are run one by one so the correction can
    be inserted at the right place; ``force_manual`` runs that path with no
    correction (used by the tests to prove both paths are identical).

    Returns ``(data, cycles, quality, chain_description)``.
    """
    extras = [k for k in _EXTRA_STEP_KEYS if s[k]]
    if not extras and not force_manual:
        result = run_pipeline(
            data,
            model=s["model"],
            butterworth_cutoff=s["butterworth_cutoff"],
            event_method=s["event_method"],
            min_cycle_duration_s=s["min_cycle_duration_s"],
            max_cycle_duration_s=s["max_cycle_duration_s"],
            n_points=s["n_points"],
            analyze=False,  # statistics are computed below, after side labelling
            direction_filter=s["direction_filter"],
            outlier_rejection=s["outlier_rejection"],
            show_progress=False,
        )
        return result["data"], result["cycles"], result["quality"], "myogait.run_pipeline"

    # Mirror of myogait.pipeline._run_steps (calibrate=False, trim_standstill=False).
    steps = []
    if s["median_spike_filter"]:
        steps.append({"type": "median", "kernel_size": s["median_kernel"]})
    steps.append({"type": "butterworth", "cutoff": s["butterworth_cutoff"], "order": 2})
    data = normalize(data, steps=steps)
    data = compute_angles(data, calibrate=False)
    if s["perspective_correction"]:
        data = apply_perspective_correction(data)
    if s["linear_detrend"]:
        data = apply_linear_detrend(data)
    data = canonicalize_angle_signs(data)
    data = detect_events(data, method=s["event_method"], trim_standstill=False,
                         min_cycle_duration=min(0.6, s["min_cycle_duration_s"]))

    def _segment():
        return segment_cycles(data, n_points=s["n_points"],
                              min_duration=s["min_cycle_duration_s"],
                              max_duration=s["max_cycle_duration_s"])

    cycles_all = _segment()
    if s["bias_correction"]:
        data = apply_hip_bias_correction(data, cycles_all)
        data = apply_knee_bias_correction(data, cycles_all)
        cycles_all = _segment()  # re-chunk the corrected continuous angles
    if s["direction_filter"]:
        cycles = filter_cycles_by_direction(data, cycles_all)
    else:
        cycles = cycles_all
        enforce_flexion_positive(cycles.get("cycles", []))
    n_outlier = 0
    if s["outlier_rejection"] and cycles.get("cycles"):
        n_before = len(cycles["cycles"])
        cycles = reject_outlier_cycles(cycles)
        n_outlier = n_before - len(cycles["cycles"])
    data["cycles_data"] = cycles
    quality = _diagnose(data, cycles_all, cycles, n_outlier=n_outlier)
    for w in quality["warnings"]:
        logger.warning("myogait: %s", w)
    chain = "myogait.run_pipeline steps" + (f" + {', '.join(extras)}" if extras else "")
    return data, cycles, quality, chain


def _swap_side_names(landmarks: dict) -> dict:
    """Return a copy of a landmark dict with LEFT_*/RIGHT_* names exchanged."""
    out = {}
    for name, value in landmarks.items():
        if name.startswith("LEFT_"):
            name = "RIGHT_" + name[5:]
        elif name.startswith("RIGHT_"):
            name = "LEFT_" + name[6:]
        out[name] = value
    return out


def _restore_anatomical_pelvis(data: dict) -> bool:
    """Give pelvic tilt/obliquity myogait's value with the true-side sign.

    myogait's pelvic tilt (``pelvis_tilt`` = ``pelvis_obliquity``, "positive =
    right side up") assumes the label layout produced by myogait's own
    mirroring, which swaps LEFT/RIGHT. This pipeline keeps the patient's true
    labels instead, which leaves every per-leg number identical but would put
    the pelvic angle near 180°. For mirrored recordings we therefore evaluate
    myogait's ``compute_angles`` on myogait's own label layout and reverse the
    sign, so that "right side up" refers to the patient's real right side.
    This equals myogait's formula applied to the original, un-mirrored video.

    Returns True when a correction was applied.
    """
    extraction = data.get("extraction") or {}
    if not (extraction.get("was_flipped")
            and extraction.get("side_label_convention") == "anatomical"):
        return False
    angle_frames = (data.get("angles") or {}).get("frames") or []
    if not angle_frames:
        return False
    mirror_layout = {
        "meta": data.get("meta", {}),
        "extraction": extraction,
        "frames": [
            {**f, "landmarks": _swap_side_names(f.get("landmarks") or {})}
            for f in data.get("frames", [])
        ],
    }
    reference = compute_angles(mirror_layout, calibrate=False)["angles"]["frames"]
    if len(reference) != len(angle_frames):
        raise RuntimeError("Pelvis reference does not align with the angle frames.")
    for af, ref in zip(angle_frames, reference):
        if af.get("frame_idx") != ref.get("frame_idx"):
            raise RuntimeError("Pelvis reference frame indices do not match.")
        for key in ("pelvis_tilt", "pelvis_obliquity"):
            v = ref.get(key)
            af[key] = None if v is None else -v
    data["angles"]["pelvis_sign_convention"] = "anatomical (positive = patient's right side up)"
    return True


def _select_side(cycles: dict, side: str) -> dict:
    """Cycles used for the statistics: all of them, or one side only."""
    if side == "both":
        return cycles
    kept = [c for c in cycles.get("cycles", []) if c.get("side") == side]
    return {"cycles": kept, "summary": summarize_cycles(kept)}


def _skipped_sides(data: dict, cycles: dict) -> List[dict]:
    """Report sides without any usable cycle, and why."""
    skipped = []
    events = data.get("events") or {}
    for side in ("left", "right"):
        if any(c.get("side") == side for c in cycles.get("cycles", [])):
            continue
        n_hs = len(events.get(f"{side}_hs", []) or [])
        skipped.append({
            "side": side,
            "reason": "insufficient_hs_events" if n_hs < 2 else "no_valid_cycles",
            "n_hs": n_hs,
        })
    return skipped


def skipped_message(s: dict) -> str:
    side = s.get("side", "unknown").capitalize()
    if s.get("reason") == "insufficient_hs_events":
        return (f"{side} side: no cycle analysis possible (fewer than 2 heel strikes detected, "
                f"n={s.get('n_hs', 0)}).")
    return (f"{side} side: {s.get('n_hs', 0)} heel strikes detected but no stride passed the "
            f"cycle filters (stride duration window, walking direction, outlier rejection). "
            f"For a slow walker, raise 'Longest stride kept'.")


def _compute_scores(cycles: dict, stratum: str) -> Tuple[dict, dict, dict]:
    if not cycles.get("cycles"):
        return {}, {}, {}
    try:
        return (gait_variable_scores(cycles, stratum=stratum),
                gait_profile_score_2d(cycles, stratum=stratum),
                sagittal_deviation_index(cycles, stratum=stratum))
    except Exception as exc:  # scores must not sink the run
        logger.warning("Gait scores could not be computed: %s", exc)
        return {}, {}, {}


def compute_dcm_gait(data: dict, settings: Dict[str, Any], force_manual: bool = False) -> dict:
    """Run the calculations on an extracted pivot dict.

    Returns ``{"data", "cycles", "stats_cycles", "stats", "quality",
    "skipped_sides", "chain", "pelvis_relabelled"}``. ``cycles`` holds both
    sides; ``stats_cycles`` the cycles the statistics were computed from.
    """
    s = settings
    data, cycles, quality, chain = _run_myogait_chain(data, s, force_manual=force_manual)
    pelvis_relabelled = _restore_anatomical_pelvis(data)

    stats_cycles = _select_side(cycles, s["visible_side"])
    stats: dict = {}
    if stats_cycles.get("cycles"):
        try:
            stats = analyze_gait(data, stats_cycles)
        except Exception as exc:  # same policy as myogait.run_pipeline
            logger.warning("analyze_gait failed: %s", exc)
    gvs, gps, sdi = _compute_scores(stats_cycles, s["normative_stratum"])
    stats["gvs"], stats["gps"], stats["sdi"] = gvs, gps, sdi

    if s["extended_angles"]:  # after the statistics: additive outputs only
        compute_extended_angles(data)
        compute_frontal_angles(data)

    return {
        "data": data,
        "cycles": cycles,
        "stats_cycles": stats_cycles,
        "stats": stats,
        "quality": quality,
        "skipped_sides": _skipped_sides(data, cycles),
        "chain": chain,
        "pelvis_relabelled": pelvis_relabelled,
    }


# =============================================================================
# Run documentation
# =============================================================================


def _score_value(score: Any, key: str) -> Any:
    return score.get(key) if isinstance(score, dict) else score


def _side_label_text(extraction: dict) -> str:
    convention = extraction.get("side_label_convention")
    direction = str(extraction.get("direction_detected") or "unknown").lower()
    if convention != "anatomical":
        return ("**Side Labels**: this pose file was extracted by standard myogait, which swaps "
                "left/right when it mirrors a left-to-right walk. For a left-to-right walk, "
                "'left' in these outputs is the patient's right. Re-extract from the video "
                "with this pipeline for true anatomical labels.")
    base = "**Side Labels**: Anatomical — left/right refer to the patient's true sides."
    if direction == "right":
        return (f"{base} Detected walking direction: left-to-right (coordinates mirrored for "
                f"analysis); the patient's **right** leg was nearest the camera, so left-side "
                f"tracking may be noisier due to occlusion.")
    if direction == "left":
        return (f"{base} Detected walking direction: right-to-left; the patient's **left** leg "
                f"was nearest the camera, so right-side tracking may be noisier due to occlusion.")
    return f"{base} Detected walking direction: unknown."


def _write_run_metadata(out_p: Path, run: dict) -> Path:
    """Save the exact run configuration and summary as JSON."""
    stats = run["stats"]
    st = stats.get("spatiotemporal", {}) or {}
    sym = stats.get("symmetry", {}) or {}
    extraction = run["data"].get("extraction", {}) or {}
    meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "myogait_version": myogait.__version__,
        "experimenter": run["experimenter"],
        "input": run["input"],
        "calculation_chain": run["chain"],
        "settings": run["settings"],
        "deviations_from_myogait_defaults": calculation_deviations(run["settings"]),
        "other_non_default_settings": [
            d for d in deviations_from_defaults(run["settings"]) if not d["changes_numbers"]
        ],
        "side_label_convention": extraction.get("side_label_convention", "myogait (swapped when mirrored)"),
        "direction_detected": extraction.get("direction_detected"),
        "was_flipped": extraction.get("was_flipped", False),
        "pelvis_sign_relabelled": run["pelvis_relabelled"],
        "effective_min_leg_vis": extraction.get("effective_min_leg_vis"),
        "results_summary": {
            "cadence_steps_per_min": st.get("cadence_steps_per_min"),
            "stride_time_mean_s": st.get("stride_time_mean_s"),
            "stride_time_std_s": st.get("stride_time_std_s"),
            "stance_pct_left": st.get("stance_pct_left"),
            "stance_pct_right": st.get("stance_pct_right"),
            "n_cycles_left": st.get("n_cycles_left"),
            "n_cycles_right": st.get("n_cycles_right"),
            "overall_symmetry_index_pct": sym.get("overall_si"),
            "gait_profile_score_gps": _score_value(stats.get("gps"), "gps_2d_overall"),
            "sagittal_deviation_index_sdi": _score_value(stats.get("sdi"), "gdi_2d_overall"),
            "clinical_flags": stats.get("pathology_flags", []),
        },
        "quality": run["quality"],
        "skipped_sides": run["skipped_sides"],
        "outputs_failed": run["outputs_failed"],
        "performance": {"elapsed_seconds": round(run["elapsed_s"], 2)},
    }
    path = out_p / "run_metadata.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    return path


def _write_run_readme(out_p: Path, run: dict) -> Path:
    """Write a human-readable summary of the run, its settings and findings."""
    stats, s = run["stats"], run["settings"]
    st = stats.get("spatiotemporal", {}) or {}
    sym = stats.get("symmetry", {}) or {}
    gps = _score_value(stats.get("gps"), "gps_2d_overall")
    sdi = _score_value(stats.get("sdi"), "gdi_2d_overall")
    flags = stats.get("pathology_flags", [])
    flags_txt = "\n".join(f"- ⚠️ **{f}**" for f in flags) if flags else "- None detected within standard thresholds."
    skipped_txt = ("\n".join(f"**Cycle Analysis Warning**: {skipped_message(x)}" for x in run["skipped_sides"])
                   if run["skipped_sides"] else "**Cycle Analysis**: Both sides segmented normally.")
    warnings_txt = "\n".join(f"- {w}" for w in run["quality"].get("warnings", [])) or "- None."
    devs = calculation_deviations(s)
    if devs:
        dev_txt = ("**Settings changed from myogait defaults (these change the numbers):**\n\n"
                   + "\n".join(f"- {d['label']} (`{d['setting']}`): `{d['value']}` (myogait default `{d['default']}`)" for d in devs))
    else:
        dev_txt = "✅ **All calculation settings are myogait defaults** (identical to `myogait.run_pipeline`)."
    settings_rows = "\n".join(
        f"| {spec.group} | {spec.label} | `{s[k]}` | `{spec.default}` |" for k, spec in SETTINGS.items()
    )
    failed_txt = ("\n\n⚠️ **Some outputs could not be generated**: " + ", ".join(run["outputs_failed"])
                  + " (see the log).") if run["outputs_failed"] else ""
    inp = run["input"]

    content = f"""# DCM Gait Analysis Run Report

**Run Timestamp**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Subject ID**: `{inp['subject_id']}`
**Subject Height**: `{inp['subject_height_m'] or 'Unspecified'} m`
**Experimenter**: `{run['experimenter'] or 'Unspecified'}`
**Input File**: `{inp['filename']}`
**Recording Plane**: `{s['plane'].capitalize()}`
**Side analysed**: `{s['visible_side']}`
**Pose Model**: `{s['model']}`
**myogait version**: `{myogait.__version__}`
**Execution Time**: `{run['elapsed_s']:.1f} s`

---

## 1. Executive Clinical Summary

| Metric | Value | Reference / Notes |
| :--- | :--- | :--- |
| **Cadence** | `{st.get('cadence_steps_per_min', 'N/A')}` steps/min | Step rate |
| **Stride Time** | `{st.get('stride_time_mean_s', 'N/A')} ± {st.get('stride_time_std_s', 'N/A')}` s | Mean ± SD |
| **Left Stance Phase** | `{st.get('stance_pct_left', 'N/A')}` % | % Gait cycle |
| **Right Stance Phase** | `{st.get('stance_pct_right', 'N/A')}` % | % Gait cycle |
| **Cycles (L / R)** | `{st.get('n_cycles_left', 0)} / {st.get('n_cycles_right', 0)}` | After myogait's cycle filters |
| **Overall Symmetry Index (SI)** | `{sym.get('overall_si', 'N/A')}` % | Normal < 10% |
| **Gait Profile Score (GPS)** | `{gps}` ° | Lower = closer to normative (`{s['normative_stratum']}`) |
| **Sagittal Deviation Index (SDI)** | `{sdi}` | Normal ~ 100 |

### Pathological Gait Flags
{flags_txt}

{skipped_txt}

{_side_label_text(run['data'].get('extraction', {}) or {})}

### myogait Quality Warnings
{warnings_txt}{failed_txt}

---

## 2. Calculation Settings

Calculation chain: `{run['chain']}`

{dev_txt}

| Group | Setting | Value used | myogait default |
| :--- | :--- | :--- | :--- |
{settings_rows}

---

## 3. Output Folder Structure

```text
{out_p.name}/
├── README.md                      <- This clinical & technical run summary
├── run_metadata.json              <- Machine-readable exact run configuration & stats
├── report/
│   └── dcm_gait_report.pdf        <- Multi-page Clinical PDF Report
├── data/
│   ├── dcm_gait_data.json         <- Full pivot JSON (landmarks, angles, events, cycles)
│   ├── dcm_gait_analysis.xlsx     <- Multi-tab Excel workbook with stats & cycle curves
│   └── csv/                       <- Time-series and cycle CSV tables
├── opensim/
│   ├── kinematics.mot             <- OpenSim joint angle kinematics (.mot format)
│   └── markers.trc                <- OpenSim marker trajectory data (.trc format)
└── plots/
    ├── summary.png                <- Clinical summary figure
    ├── angles.png                 <- Joint angle time-series plot
    ├── events.png                 <- Event detection plot (heel-strike & toe-off)
    ├── cycles_left.png            <- Normalized left gait cycle curves (0-100%)
    ├── cycles_right.png           <- Normalized right gait cycle curves (0-100%)
    ├── skeleton_overlay.mp4       <- Video with pose overlay
    └── stickfigure_video.mp4      <- Anonymized stick-figure video
```

---
*Generated automatically by the DCM Clinical Gait Analysis Pipeline (myogait {myogait.__version__}).*
"""
    path = out_p / "README.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


# =============================================================================
# Reporting mask (display only)
# =============================================================================


def _mask_for_display(data: dict, cycles: dict, side: str) -> Tuple[dict, dict]:
    """Hide the other leg in plots, videos and reports. Numbers are untouched."""
    if side == "both":
        return data, cycles
    other_short, other_long = ("R", "right") if side == "left" else ("L", "left")
    data = dict(data)
    if "angles" in data and "frames" in data["angles"]:
        data["angles"] = dict(data["angles"])
        data["angles"]["frames"] = [
            {k: v for k, v in af.items()
             if not (f"_{other_short}" in k or k.endswith(f"_{other_short.lower()}"))}
            for af in data["angles"]["frames"]
        ]
    data["events"] = {k: v for k, v in (data.get("events") or {}).items() if other_long not in k}
    cycles = dict(cycles)
    cycles["cycles"] = [c for c in cycles.get("cycles", []) if c.get("side") == side]
    if "summary" in cycles:
        cycles["summary"] = {k: v for k, v in cycles["summary"].items() if k != other_long}
    return data, cycles


# =============================================================================
# Main entry point
# =============================================================================


def run_dcm_pipeline(
    video_path: Optional[str] = None,
    input_json_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    subject_id: Optional[str] = None,
    subject_height_m: Optional[float] = None,
    experimenter: str = "Unknown",
    settings: Optional[Dict[str, Any]] = None,
    generate_pdf: bool = True,
    generate_excel: bool = True,
    generate_csv: bool = True,
    generate_opensim: bool = True,
    generate_plots: bool = True,
    generate_videos: bool = True,
    **setting_overrides: Any,
) -> Dict[str, Any]:
    """Analyse one video (or pre-extracted pose JSON) and write all outputs.

    Settings come from ``settings`` and/or keyword overrides (any key of
    :data:`SETTINGS`, e.g. ``max_cycle_duration_s=2.5``); everything not
    given uses the myogait default.

    Returns a dict with ``data``, ``cycles``, ``stats``, ``quality``,
    ``settings`` and ``output_dir``.
    """
    s = resolve_settings({**(settings or {}), **setting_overrides})

    if video_path:
        src = Path(video_path)
        if not src.exists():
            raise FileNotFoundError(f"Input video not found: {video_path}")
    elif input_json_path:
        src = Path(input_json_path)
        if not src.exists():
            raise FileNotFoundError(f"Input JSON not found: {input_json_path}")
    else:
        raise ValueError("Must provide either video_path or input_json_path.")
    subject_slug = subject_id or src.stem

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_p = Path(output_dir or "./Output") / f"{subject_slug}_{s['plane']}_{s['model']}_{timestamp_str}"
    plots_p, report_p, data_p = out_p / "plots", out_p / "report", out_p / "data"
    csv_p, opensim_p = data_p / "csv", out_p / "opensim"
    for p in (plots_p, report_p, data_p, csv_p, opensim_p):
        p.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    devs = calculation_deviations(s)
    logger.info("=" * 60)
    logger.info(f"Starting DCM Pipeline for: {src}")
    logger.info(f"Pose model: {s['model']} | Plane: {s['plane']} | myogait {myogait.__version__}")
    if devs:
        logger.info("Settings changed from myogait defaults: "
                    + ", ".join(f"{d['setting']}={d['value']}" for d in devs))
    else:
        logger.info("All calculation settings are myogait defaults.")
    logger.info("=" * 60)

    # 1. Pose extraction ------------------------------------------------------
    logger.info("[1/4] Extracting pose landmarks (or loading from JSON)...")
    t0 = time.time()
    if video_path:
        data = extract(
            str(src),
            model=s["model"],
            flip_if_right=s["flip_if_right"],
            correct_inversions=s["correct_inversions"],
            edge_margin=s["edge_margin"],
            min_leg_vis=s["min_leg_vis"],
            adaptive_leg_vis=s["adaptive_leg_vis"],
        )
        n_frames = len(data.get("frames", []))
        logger.info(f"      Extracted {n_frames} frames in {time.time() - t0:.1f}s")
    else:
        data = load_json(str(src))
        logger.info(f"      Loaded pre-extracted data in {time.time() - t0:.1f}s")
    data.setdefault("meta", {})["plane"] = s["plane"]

    if subject_id or subject_height_m:
        data = set_subject(data, subject_id=subject_id or "DCM_Subject", height_m=subject_height_m)

    # 2. myogait calculations -------------------------------------------------
    logger.info("[2/4] Running myogait calculations (filtering, angles, events, cycles, statistics)...")
    calc = compute_dcm_gait(data, s)
    data, cycles, stats = calc["data"], calc["cycles"], calc["stats"]
    logger.info(f"      Chain: {calc['chain']}")
    logger.info(f"      {len(cycles.get('cycles', []))} gait cycles kept.")
    for skipped in calc["skipped_sides"]:
        logger.warning("      " + skipped_message(skipped))
    st = stats.get("spatiotemporal", {}) or {}
    logger.info(f"      Cadence: {st.get('cadence_steps_per_min', 'N/A')} steps/min | "
                f"Stride time: {st.get('stride_time_mean_s', 'N/A')} s")
    for flag in stats.get("pathology_flags", []):
        logger.info(f"      Clinical flag: {flag}")

    json_path = data_p / "dcm_gait_data.json"
    save_json(data, str(json_path))

    # 3. Outputs ----------------------------------------------------------------
    logger.info("[3/4] Generating clinical outputs and exports...")
    view_data, view_cycles = _mask_for_display(data, cycles, s["visible_side"])
    failed: List[str] = []

    def _attempt(name: str, fn) -> None:
        try:
            fn()
        except Exception as exc:
            failed.append(name)
            logger.warning(f"      Could not generate {name}: {exc}")

    if generate_plots:
        def _save(fig, name):
            fig.savefig(plots_p / name, dpi=150, bbox_inches="tight")
            plt.close(fig)
        _attempt("summary plot", lambda: _save(plot_summary(view_data, view_cycles, stats), "summary.png"))
        _attempt("angles plot", lambda: _save(plot_angles(view_data), "angles.png"))
        _attempt("events plot", lambda: _save(plot_events(view_data), "events.png"))
        for side in ("left", "right"):
            if s["visible_side"] in ("both", side) and any(c.get("side") == side for c in cycles.get("cycles", [])):
                _attempt(f"{side} cycles plot",
                         lambda side=side: _save(plot_cycles(view_cycles, side=side), f"cycles_{side}.png"))

    if generate_videos:
        from myogait.video import render_skeleton_video, render_stickfigure_video
        if video_path:
            _attempt("skeleton overlay video", lambda: render_skeleton_video(
                video_path=str(src), data=view_data, output_path=str(plots_p / "skeleton_overlay.mp4"),
                show_angles=True, show_events=True, visible_side=s["visible_side"]))
        _attempt("stick-figure video", lambda: render_stickfigure_video(
            data=view_data, output_path=str(plots_p / "stickfigure_video.mp4"),
            show_angles=True, show_events=True, visible_side=s["visible_side"]))

    if generate_pdf:
        _attempt("PDF report", lambda: generate_report(
            view_data, view_cycles, stats, str(report_p / "dcm_gait_report.pdf"),
            language=s["report_language"]))
    if generate_excel:
        _attempt("Excel workbook", lambda: export_excel(
            view_data, str(data_p / "dcm_gait_analysis.xlsx"), view_cycles, stats))
    if generate_csv:
        _attempt("CSV tables", lambda: export_csv(view_data, str(csv_p), view_cycles, stats))
    if generate_opensim:
        _attempt("OpenSim .mot", lambda: export_mot(view_data, str(opensim_p / "kinematics.mot")))
        _attempt("OpenSim .trc", lambda: export_trc(view_data, str(opensim_p / "markers.trc")))

    # 4. Run documentation -------------------------------------------------------
    logger.info("[4/4] Writing README.md and run_metadata.json...")
    run = {
        "settings": s,
        "experimenter": experimenter,
        "input": {
            "path": str(src.resolve()),
            "filename": src.name,
            "type": "video" if video_path else "json",
            "subject_id": subject_id or "Unspecified",
            "subject_height_m": subject_height_m,
        },
        "data": data,
        "stats": stats,
        "quality": calc["quality"],
        "skipped_sides": calc["skipped_sides"],
        "chain": calc["chain"],
        "pelvis_relabelled": calc["pelvis_relabelled"],
        "outputs_failed": failed,
        "elapsed_s": time.time() - t_start,
    }
    _write_run_metadata(out_p, run)
    _write_run_readme(out_p, run)

    logger.info("=" * 60)
    status = "with missing outputs: " + ", ".join(failed) if failed else "successfully"
    logger.info(f"DCM Pipeline completed {status} in {run['elapsed_s']:.1f}s")
    logger.info(f"Outputs written to: {out_p.resolve()}")
    logger.info("=" * 60)

    return {
        "data": data,
        "cycles": cycles,
        "stats": stats,
        "quality": calc["quality"],
        "skipped_sides": calc["skipped_sides"],
        "settings": s,
        "outputs_failed": failed,
        "output_dir": str(out_p.resolve()),
    }


# =============================================================================
# Command line
# =============================================================================


def _add_setting_arguments(parser: argparse.ArgumentParser) -> None:
    cli_aliases = {"butterworth_cutoff": ["--cutoff"], "event_method": ["--events-method"],
                   "report_language": ["--language"]}
    for group_name in GROUPS:
        group = parser.add_argument_group(group_name)
        for spec in SETTINGS_LIST:
            if spec.group != group_name:
                continue
            flags = ["--" + spec.key.replace("_", "-")] + cli_aliases.get(spec.key, [])
            helptext = f"{spec.help} (default: {spec.default})"
            if spec.kind == "bool":
                group.add_argument(*flags, dest=spec.key, action=argparse.BooleanOptionalAction,
                                   default=None, help=helptext)
            else:
                typ = {"float": float, "int": int}.get(spec.kind, str)
                group.add_argument(*flags, dest=spec.key, type=typ, default=None,
                                   choices=spec.choices or None, help=helptext)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="DCM clinical gait analysis on sagittal or coronal walking videos. "
                    "Every calculation setting defaults to myogait's validated value.",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=str, help="Path to one video file.")
    src.add_argument("--json", type=str, help="Path to a pre-extracted myogait pose JSON.")
    src.add_argument("--batch-dir", type=str, help="Folder of videos to process one by one.")
    src.add_argument("--list-settings", action="store_true", help="Print every setting and exit.")
    parser.add_argument("--output-dir", type=str, default="./Output", help="Base output folder.")
    parser.add_argument("--subject-id", type=str, default=None, help="Subject ID (e.g. DCM_001).")
    parser.add_argument("--height", type=float, default=None, help="Subject height in metres.")
    parser.add_argument("--experimenter", type=str, default="Unknown", help="Who ran the analysis.")
    for name in ("pdf", "excel", "csv", "opensim", "plots", "videos"):
        parser.add_argument(f"--no-{name}", action="store_true", help=f"Skip the {name} output.")
    _add_setting_arguments(parser)
    args = parser.parse_args(argv)

    if args.list_settings:
        for group_name in GROUPS:
            print(f"\n== {group_name}")
            for spec in SETTINGS_LIST:
                if spec.group == group_name:
                    print(f"  --{spec.key.replace('_', '-'):24s} default={spec.default!r:18} {spec.label}")
        return

    overrides = {k: getattr(args, k) for k in SETTINGS if getattr(args, k) is not None}
    common = dict(
        output_dir=args.output_dir, subject_height_m=args.height, experimenter=args.experimenter,
        settings=overrides, generate_pdf=not args.no_pdf, generate_excel=not args.no_excel,
        generate_csv=not args.no_csv, generate_opensim=not args.no_opensim,
        generate_plots=not args.no_plots, generate_videos=not args.no_videos,
    )
    if args.video or args.json:
        run_dcm_pipeline(video_path=args.video, input_json_path=args.json,
                         subject_id=args.subject_id, **common)
        return

    batch_p = Path(args.batch_dir)
    videos = sorted(p for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv") for p in batch_p.glob(ext))
    if not videos:
        logger.error(f"No video files found in {args.batch_dir}")
        sys.exit(1)
    logger.info(f"Found {len(videos)} videos in {args.batch_dir} for batch processing.")
    for vid in videos:
        try:
            run_dcm_pipeline(video_path=str(vid), subject_id=args.subject_id or vid.stem, **common)
        except Exception as e:
            logger.error(f"Error processing video {vid.name}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
