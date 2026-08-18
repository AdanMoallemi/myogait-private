"""Coverage-focused tests for extract helper paths."""

import numpy as np
import pytest


def test_goliath_to_mediapipe_maps_and_builds_foot_index():
    from myogait.extract import _goliath_to_mediapipe
    from myogait.constants import MP_NAME_TO_INDEX

    g308 = np.full((308, 3), np.nan, dtype=float)
    # Directly mapped point (LEFT_HIP via GOLIATH_TO_MP)
    g308[9] = [0.4, 0.5, 0.9]
    # Left foot index midpoint from big/small toe (15, 16)
    g308[15] = [0.2, 0.8, 0.7]
    g308[16] = [0.4, 0.9, 0.6]

    mp = _goliath_to_mediapipe(g308)
    assert mp.shape == (33, 3)
    assert np.isclose(mp[MP_NAME_TO_INDEX["LEFT_HIP"], 0], 0.4)
    lf = mp[MP_NAME_TO_INDEX["LEFT_FOOT_INDEX"]]
    assert np.isclose(lf[0], 0.3)
    assert np.isclose(lf[1], 0.85)
    assert np.isclose(lf[2], 0.6)


def test_enrich_foot_landmarks_from_goliath308():
    from myogait.extract import _enrich_foot_landmarks

    frame = {
        "landmarks": {"LEFT_ANKLE": {"x": 0.2, "y": 0.8, "visibility": 1.0}},
        "goliath308": [[np.nan, np.nan, np.nan] for _ in range(308)],
    }
    # left big/small toe + heel
    frame["goliath308"][15] = [0.2, 0.8, 0.9]
    frame["goliath308"][16] = [0.4, 0.9, 0.8]
    frame["goliath308"][17] = [0.3, 0.85, 0.7]

    _enrich_foot_landmarks(frame)

    assert frame["foot_landmarks_source"] == "detected"
    assert "LEFT_BIG_TOE" in frame["landmarks"]
    assert "LEFT_HEEL" in frame["landmarks"]
    assert "LEFT_FOOT_INDEX" in frame["landmarks"]
    assert frame["landmarks"]["LEFT_FOOT_INDEX"]["x"] == pytest.approx(0.3)


def test_enrich_foot_landmarks_from_wholebody133():
    from myogait.extract import _enrich_foot_landmarks

    frame = {
        "landmarks": {"LEFT_ANKLE": {"x": 0.3, "y": 0.8, "visibility": 1.0}},
        "wholebody133": [[np.nan, np.nan, np.nan] for _ in range(133)],
    }
    # RTMW foot indices: left big toe, small toe, heel
    frame["wholebody133"][17] = [0.3, 0.8, 0.9]
    frame["wholebody133"][18] = [0.4, 0.82, 0.8]
    frame["wholebody133"][19] = [0.35, 0.84, 0.7]

    _enrich_foot_landmarks(frame)

    assert frame["foot_landmarks_source"] == "detected"
    assert "LEFT_BIG_TOE" in frame["landmarks"]
    assert "LEFT_SMALL_TOE" in frame["landmarks"]
    assert "LEFT_HEEL" in frame["landmarks"]


def test_flip_auxiliary_mirrors_x():
    from myogait.extract import _flip_auxiliary

    names = ["left_eye", "right_eye", "nose"]
    aux = np.array(
        [
            [0.1, 0.2, 1.0],
            [0.9, 0.2, 1.0],
            [0.5, 0.3, 1.0],
        ],
        dtype=float,
    )

    flipped = _flip_auxiliary(aux, names)
    # X coordinate mirrored, anatomical indices preserved
    assert flipped[0, 0] == pytest.approx(1.0 - aux[0, 0])
    assert flipped[1, 0] == pytest.approx(1.0 - aux[1, 0])
    assert flipped[2, 0] == pytest.approx(0.5)


def test_extract_missing_file_raises_file_not_found():
    from myogait.extract import extract

    with pytest.raises(FileNotFoundError):
        extract("/definitely/not/here.mp4")


def test_side_label_convention_constant():
    from myogait.extract import SIDE_LABEL_CONVENTION, _flip_landmarks

    assert SIDE_LABEL_CONVENTION == "anatomical"

    # Test that _flip_landmarks mirrors x coordinate without swapping indices
    lm = np.zeros((33, 3), dtype=float)
    lm[11] = [0.2, 0.5, 1.0]  # LEFT_SHOULDER
    lm[12] = [0.8, 0.5, 1.0]  # RIGHT_SHOULDER

    flipped = _flip_landmarks(lm)
    assert flipped[11, 0] == pytest.approx(0.8)
    assert flipped[12, 0] == pytest.approx(0.2)


def test_confidence_and_leg_confidence_distinct_means():
    """Verify confidence uses all landmarks while leg_confidence uses leg landmarks."""
    from myogait.extract import _compute_confidences
    from myogait.constants import MP_NAME_TO_INDEX

    leg_indices = [
        MP_NAME_TO_INDEX["LEFT_HIP"],
        MP_NAME_TO_INDEX["RIGHT_HIP"],
        MP_NAME_TO_INDEX["LEFT_KNEE"],
        MP_NAME_TO_INDEX["RIGHT_KNEE"],
        MP_NAME_TO_INDEX["LEFT_ANKLE"],
        MP_NAME_TO_INDEX["RIGHT_ANKLE"],
    ]

    lm = np.full((33, 3), 0.5, dtype=float)
    # Set non-leg landmarks to high visibility (0.9)
    lm[:, 2] = 0.9
    # Set leg landmarks to low visibility (0.3)
    for idx in leg_indices:
        lm[idx, 2] = 0.3

    conf, leg_conf = _compute_confidences(lm, leg_indices)

    assert isinstance(conf, float)
    assert isinstance(leg_conf, float)

    # All-landmark mean: (27 * 0.9 + 6 * 0.3) / 33 = 26.1 / 33
    expected_all_mean = float(np.mean(lm[:, 2]))
    assert conf == pytest.approx(expected_all_mean)
    assert leg_conf == pytest.approx(0.3)
    assert conf != leg_conf


def test_confidence_and_leg_confidence_no_valid_legs_fallback():
    """When leg landmarks are all NaN, leg_confidence falls back to all-landmark confidence."""
    from myogait.extract import _compute_confidences
    from myogait.constants import MP_NAME_TO_INDEX

    leg_indices = [
        MP_NAME_TO_INDEX["LEFT_HIP"],
        MP_NAME_TO_INDEX["RIGHT_HIP"],
        MP_NAME_TO_INDEX["LEFT_KNEE"],
        MP_NAME_TO_INDEX["RIGHT_KNEE"],
        MP_NAME_TO_INDEX["LEFT_ANKLE"],
        MP_NAME_TO_INDEX["RIGHT_ANKLE"],
    ]

    lm = np.full((33, 3), 0.5, dtype=float)
    lm[:, 2] = 0.8
    # Leg landmarks are NaN
    for idx in leg_indices:
        lm[idx, 2] = np.nan

    conf, leg_conf = _compute_confidences(lm, leg_indices)

    assert isinstance(conf, float)
    assert isinstance(leg_conf, float)
    assert conf == pytest.approx(0.8)
    assert leg_conf == pytest.approx(0.8)  # Fell back to all-landmark mean


def test_confidence_and_leg_confidence_empty_and_none():
    """All-NaN and None inputs return 0.0 for both confidence fields."""
    from myogait.extract import _compute_confidences

    # All NaN
    lm = np.full((33, 3), np.nan, dtype=float)
    conf, leg_conf = _compute_confidences(lm)
    assert conf == 0.0 and isinstance(conf, float)
    assert leg_conf == 0.0 and isinstance(leg_conf, float)

    # None
    conf_none, leg_conf_none = _compute_confidences(None)
    assert conf_none == 0.0 and isinstance(conf_none, float)
    assert leg_conf_none == 0.0 and isinstance(leg_conf_none, float)


