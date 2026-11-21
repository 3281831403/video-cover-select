"""Contract tests for cover.scoring.

The initial stubs raise ``NotImplementedError``, so every test below fails.
Tests are intentionally read-only with respect to the implementation: fix
``cover/scoring.py`` rather than changing this file.
"""

import pytest

from cover.scoring import (
    CoverError,
    brightness_score,
    composition,
    contrast_score,
    cover_batch,
    face_proxy,
    pick_best,
    sharpness,
    total_score,
)

ABS = 1e-6

WEIGHTS = {
    "sharpness": 0.35,
    "brightness": 0.20,
    "contrast": 0.15,
    "face": 0.15,
    "composition": 0.15,
}

ZERO_WEIGHTS = {
    "sharpness": 0.0,
    "brightness": 0.0,
    "contrast": 0.0,
    "face": 0.0,
    "composition": 0.0,
}


def flat_frame(rows=8, cols=8, value=0.5):
    """Return a constant grayscale frame."""
    return [[float(value) for _ in range(cols)] for _ in range(rows)]


def checker_frame(rows=8, cols=8):
    """Return a high-detail 0/1 checkerboard whose mean is 0.5."""
    return [[float((row + col) % 2) for col in range(cols)] for row in range(rows)]


def vertical_edge_frame(width=12, height=12, boundary=4):
    """Return a vertical step edge with the given column boundary."""
    return [
        [0.0 if col < boundary else 1.0 for col in range(width)]
        for _ in range(height)
    ]


def candidate(idx, frame):
    """Build a candidate mapping with the public candidate shape."""
    return {"idx": idx, "frame": frame}


# ---------------------------------------------------------------------------
# One test per stated defect.
# ---------------------------------------------------------------------------


def test_sharpness_uses_variance_not_mean():
    """Defect 1: a mean-based score ties a flat and a detailed frame."""
    blurred = flat_frame(value=0.5)
    detailed = checker_frame()

    blurred_score = sharpness(blurred)
    detailed_score = sharpness(detailed)

    assert blurred_score == pytest.approx(0.0, abs=ABS)
    assert 0.0 <= detailed_score <= 1.0
    assert detailed_score > blurred_score + ABS


def test_brightness_score_is_normalized():
    """Defect 2: absolute brightness error was returned without normalizing."""
    mid = flat_frame(value=0.5)
    black = flat_frame(value=0.0)
    white = flat_frame(value=1.0)

    assert brightness_score(mid, 0.5) == pytest.approx(1.0, abs=ABS)
    assert brightness_score(black, 0.5) == pytest.approx(0.0, abs=ABS)
    assert brightness_score(white, 0.5) == pytest.approx(0.0, abs=ABS)

    for frame in (black, mid, white):
        score = brightness_score(frame, 0.9)
        assert 0.0 <= score <= 1.0

    assert brightness_score(white, 0.9) > brightness_score(black, 0.9)


def test_contrast_uses_normalized_full_range():
    """Defect 3: max-min was scaled like 8-bit data and dark frames won."""
    dark_low_range = [
        [0.00, 0.03, 0.06],
        [0.02, 0.05, 0.08],
    ]
    full_range = [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
    ]

    dark_score = contrast_score(dark_low_range)
    full_score = contrast_score(full_range)

    assert 0.0 <= dark_score <= 1.0
    assert full_score == pytest.approx(1.0, abs=ABS)
    assert full_score > dark_score + ABS
    assert contrast_score(flat_frame()) == pytest.approx(0.0, abs=ABS)


def test_face_proxy_center_weight_is_normalized():
    """Defect 4: a fixed 1.5 multiplier was not divided by total weight."""
    box = (0.25, 0.25, 0.75, 0.75)
    white = flat_frame(value=1.0)

    assert face_proxy(white, box) == pytest.approx(1.0, abs=ABS)

    center_bright = [[0.0 for _ in range(8)] for _ in range(8)]
    for row in range(2, 6):
        for col in range(2, 6):
            center_bright[row][col] = 1.0

    plain_mean = sum(sum(row) for row in center_bright) / 64.0
    proxy = face_proxy(center_bright, box)

    assert 0.0 <= proxy <= 1.0
    assert proxy > plain_mean + ABS


def test_composition_uses_vertical_thirds():
    """Defect 5: only horizontal third lines were considered."""
    on_vertical_third = vertical_edge_frame(boundary=4)
    at_center = vertical_edge_frame(boundary=6)

    on_third_score = composition(on_vertical_third)
    center_score = composition(at_center)

    assert 0.0 <= on_third_score <= 1.0
    assert 0.0 <= center_score <= 1.0
    assert on_third_score > center_score + ABS


def test_total_score_weight_normalization():
    """Defect 6: raw weights made the total exceed 1."""
    detailed = checker_frame()

    base = total_score(detailed, WEIGHTS)
    tripled = {key: value * 3.0 for key, value in WEIGHTS.items()}

    assert 0.0 <= base <= 1.0
    assert base > 0.0
    assert total_score(detailed, tripled) == pytest.approx(base, abs=ABS)


def test_pick_best_prefers_earlier_on_tie():
    """Defect 7: ties were resolved by keeping the last candidate."""
    frame = checker_frame()
    candidates = [
        candidate(10, frame),
        candidate(20, frame),
        candidate(30, frame),
    ]

    assert pick_best(candidates) == 10


def test_pick_best_empty_candidates_raises():
    """Defect 8: an empty candidate list returned -1 and indexed out of range."""
    with pytest.raises(CoverError):
        pick_best([])


def test_cover_batch_does_not_share_scores_between_videos():
    """Defect 9: a shared score array mixed candidates across videos."""
    detailed = checker_frame()
    black = flat_frame(value=0.0)

    videos = [
        {
            "id": "a",
            "candidates": [candidate(2, black), candidate(5, detailed)],
        },
        {
            "id": "b",
            "candidates": [
                candidate(7, detailed),
                candidate(8, black),
                candidate(9, black),
            ],
        },
        {
            "id": "c",
            "candidates": [
                candidate(4, black),
                candidate(6, black),
                candidate(3, detailed),
            ],
        },
    ]

    result = cover_batch(videos, {})

    assert result == {"a": 5, "b": 7, "c": 3}


def test_total_score_rejects_all_zero_weights():
    """Defect 10: all-zero weights divided by zero instead of raising."""
    with pytest.raises(CoverError):
        total_score(flat_frame(), ZERO_WEIGHTS)


# ---------------------------------------------------------------------------
# Five additional combined/contract tests.
# ---------------------------------------------------------------------------


def test_all_black_frame_has_zero_total_score():
    """A fully black frame contributes zero to every score component."""
    black = flat_frame(rows=10, cols=10, value=0.0)

    assert total_score(black, WEIGHTS) == pytest.approx(0.0, abs=ABS)


def test_sharp_frame_beats_blurred_frame():
    """A detailed checkerboard must rank above a constant gray frame."""
    detailed = checker_frame()
    blurred = flat_frame(value=0.5)

    assert sharpness(detailed) > sharpness(blurred) + ABS
    assert total_score(detailed, WEIGHTS) > total_score(blurred, WEIGHTS) + ABS


def test_empty_candidates_raise_cover_error_in_batch():
    """An empty candidate list invalidates the whole batch."""
    videos = [
        {
            "id": "good",
            "candidates": [candidate(1, checker_frame())],
        },
        {
            "id": "bad",
            "candidates": [],
        },
    ]

    with pytest.raises(CoverError):
        cover_batch(videos, {})


def test_all_zero_weights_raise_cover_error_in_batch():
    """Batch mode validates configured weights instead of dividing by zero."""
    videos = [
        {
            "id": "v1",
            "candidates": [candidate(1, checker_frame())],
        }
    ]

    with pytest.raises(CoverError):
        cover_batch(videos, {"weights": ZERO_WEIGHTS})


def test_batch_preserves_video_order():
    """A successful batch returns every id in the original input order."""
    videos = [
        {
            "id": "alpha",
            "candidates": [candidate(11, checker_frame())],
        },
        {
            "id": "beta",
            "candidates": [candidate(22, flat_frame(value=0.0))],
        },
        {
            "id": "gamma",
            "candidates": [candidate(33, flat_frame(value=1.0))],
        },
    ]

    result = cover_batch(videos, {})

    assert list(result.keys()) == ["alpha", "beta", "gamma"]
    assert result == {"alpha": 11, "beta": 22, "gamma": 33}
