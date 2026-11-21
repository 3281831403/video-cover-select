"""Frame scoring primitives for automatic video cover selection.

A frame is a non-empty, rectangular two-dimensional grayscale array
(``Sequence[Sequence[float]]``) whose finite values lie in ``[0.0, 1.0]``.
Unless a function says otherwise, every public score is a finite float in
``[0.0, 1.0]``.  Values that are expected to tie are compared with a tolerance
of ``1e-6``.

The module deliberately contains no I/O and no third-party dependencies.
Invalid input (malformed frame, bad target, bad box, unknown weights, empty
candidate list, and so on) must raise :class:`CoverError`, which is the only
exception type callers need to handle.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "CoverError",
    "DEFAULT_BRIGHTNESS_TARGET",
    "DEFAULT_WEIGHTS",
    "sharpness",
    "brightness_score",
    "contrast_score",
    "face_proxy",
    "composition",
    "total_score",
    "pick_best",
    "cover_batch",
]


class CoverError(ValueError):
    """Raised when cover scoring receives invalid input."""


#: Brightness target used by :func:`total_score` when no per-call target exists.
DEFAULT_BRIGHTNESS_TARGET = 0.5

#: Reference weights used by :func:`pick_best`.  They sum to 1.0, although
#: every public scoring function also accepts unnormalised non-negative weights
#: and normalises them itself.
DEFAULT_WEIGHTS = {
    "sharpness": 0.35,
    "brightness": 0.20,
    "contrast": 0.15,
    "face": 0.15,
    "composition": 0.15,
}

#: Weight applied to pixels inside the face-proxy box, before normalisation.
FACE_CENTER_WEIGHT = 1.5

#: Tolerance used when treating two scores as a tie.
_TOL = 1e-6

_WEIGHT_KEYS = ("sharpness", "brightness", "contrast", "face", "composition")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _is_real_number(value: Any) -> bool:
    """Return whether ``value`` is a real, finite, non-boolean number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _validate_frame(frame: Any) -> list[list[float]]:
    """Validate ``frame`` and return a plain float copy of it.

    A valid frame is a non-empty rectangular two-dimensional sequence whose
    finite, non-boolean numeric values all lie in ``[0.0, 1.0]``.
    """
    if isinstance(frame, (str, bytes, Mapping)) or not isinstance(
        frame, Sequence
    ):
        raise CoverError("frame must be a non-empty 2D sequence")

    rows = len(frame)
    if rows == 0:
        raise CoverError("frame must have at least one row")

    width: int | None = None
    validated: list[list[float]] = []
    for row in frame:
        if isinstance(row, (str, bytes, Mapping)) or not isinstance(
            row, Sequence
        ):
            raise CoverError("every frame row must be a sequence")
        if width is None:
            width = len(row)
            if width == 0:
                raise CoverError("frame must have at least one column")
        elif len(row) != width:
            raise CoverError("frame rows must all have the same length")

        validated_row: list[float] = []
        for value in row:
            if not _is_real_number(value):
                raise CoverError(
                    "frame values must be finite numbers in [0.0, 1.0]"
                )
            number = float(value)
            if number < 0.0 or number > 1.0:
                raise CoverError(
                    "frame values must lie in [0.0, 1.0]"
                )
            validated_row.append(number)
        validated.append(validated_row)

    return validated


def _validate_weights(weights: Any) -> dict[str, float]:
    """Validate and normalise a weight mapping.

    Missing keys count as zero, unknown keys or non-finite/negative values
    raise :class:`CoverError`, and an all-zero mapping is rejected so callers
    never divide by zero.
    """
    if not isinstance(weights, Mapping):
        raise CoverError("weights must be a mapping")

    validated: dict[str, float] = {}
    total = 0.0
    for key in _WEIGHT_KEYS:
        if key not in weights:
            validated[key] = 0.0
            continue
        value = weights[key]
        if not _is_real_number(value):
            raise CoverError(f"weight {key!r} must be a finite number")
        number = float(value)
        if number < 0.0:
            raise CoverError(f"weight {key!r} must be non-negative")
        validated[key] = number
        total += number

    if len(weights) != len(_WEIGHT_KEYS):
        for key in weights:
            if key not in validated:
                raise CoverError(f"unknown weight key: {key!r}")

    if total <= 0.0:
        raise CoverError("weights must have a strictly positive sum")

    return validated


def _clamp_unit(value: float) -> float:
    """Clamp tiny floating-point excursions back into ``[0.0, 1.0]``."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


# ---------------------------------------------------------------------------
# Component scores
# ---------------------------------------------------------------------------


def sharpness(frame: Sequence[Sequence[float]]) -> float:
    """Return a normalized Laplacian-variance sharpness score for ``frame``.

    The four-neighbour discrete Laplacian measures local detail rather than
    mean brightness.  Each interior response lies in ``[-4, 4]`` because the
    input values are in ``[0, 1]``, so the population variance of the responses
    is bounded by ``16``; dividing by ``16`` therefore normalizes the score to
    ``[0.0, 1.0]``.  A constant frame has zero variance and scores exactly
    ``0.0``.
    """
    grid = _validate_frame(frame)
    rows = len(grid)
    cols = len(grid[0])

    # Frames without a 2x2 interior cannot carry a four-neighbour interior.
    if rows < 3 or cols < 3:
        return 0.0

    responses: list[float] = []
    for i in range(1, rows - 1):
        upper = grid[i - 1]
        current = grid[i]
        lower = grid[i + 1]
        for j in range(1, cols - 1):
            laplacian = (
                upper[j]
                + lower[j]
                + current[j - 1]
                + current[j + 1]
                - 4.0 * current[j]
            )
            responses.append(laplacian)

    count = len(responses)
    mean = sum(responses) / count
    variance = sum((value - mean) ** 2 for value in responses) / count
    return _clamp_unit(variance / 16.0)


def brightness_score(
    frame: Sequence[Sequence[float]], target: float
) -> float:
    """Return how close ``frame``'s mean brightness is to ``target``.

    The absolute error is divided by the distance from ``target`` to the
    farthest brightness endpoint, which keeps the score inside ``[0.0, 1.0]``
    and makes ``mean == target`` score exactly ``1.0``.
    """
    grid = _validate_frame(frame)
    if not _is_real_number(target):
        raise CoverError("brightness target must be a finite number")
    target_value = float(target)
    if target_value < 0.0 or target_value > 1.0:
        raise CoverError("brightness target must lie in [0.0, 1.0]")

    pixel_count = len(grid) * len(grid[0])
    total = sum(sum(row) for row in grid)
    mean = total / pixel_count

    denominator = max(target_value, 1.0 - target_value)
    score = 1.0 - abs(mean - target_value) / denominator
    return _clamp_unit(score)


def contrast_score(frame: Sequence[Sequence[float]]) -> float:
    """Return a normalized dynamic-range contrast score for ``frame``.

    The frame already uses the unit grayscale scale (an 8-bit value divided by
    255), so ``max - min`` is itself the normalized span and must never be
    multiplied by 255 again.
    """
    grid = _validate_frame(frame)

    minimum = math.inf
    maximum = -math.inf
    for row in grid:
        row_min = min(row)
        row_max = max(row)
        if row_min < minimum:
            minimum = row_min
        if row_max > maximum:
            maximum = row_max

    return _clamp_unit(maximum - minimum)


def face_proxy(
    frame: Sequence[Sequence[float]],
    box: Sequence[float],
) -> float:
    """Return a normalized center-weighting proxy for a suspected face.

    Pixels inside ``box`` receive :data:`FACE_CENTER_WEIGHT`; every other
    pixel receives ``1.0``.  The weighted sum is divided by the total weight,
    so an all-``1.0`` frame still scores exactly ``1.0``.
    """
    grid = _validate_frame(frame)

    if isinstance(box, (str, bytes, Mapping)) or not isinstance(
        box, Sequence
    ) or len(box) != 4:
        raise CoverError("box must be a sequence of four coordinates")
    coordinates = tuple(box)
    if not all(_is_real_number(value) for value in coordinates):
        raise CoverError("box coordinates must be finite numbers")
    left, top, right, bottom = (float(value) for value in coordinates)

    if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
        raise CoverError(
            "box must satisfy 0 <= left < right <= 1 and "
            "0 <= top < bottom <= 1"
        )

    rows = len(grid)
    cols = len(grid[0])
    weighted_sum = 0.0
    total_weight = 0.0
    selected = 0
    for i, row in enumerate(grid):
        # Pixel centers use normalized coordinates (i + 0.5) / dimension.
        y = (i + 0.5) / rows
        in_vertical_band = top <= y < bottom
        for j, value in enumerate(row):
            x = (j + 0.5) / cols
            if in_vertical_band and left <= x < right:
                weight = FACE_CENTER_WEIGHT
                selected += 1
            else:
                weight = 1.0
            weighted_sum += value * weight
            total_weight += weight

    if selected == 0:
        raise CoverError("box must select at least one pixel center")

    return _clamp_unit(weighted_sum / total_weight)


def _line_weight(position: float) -> float:
    """Weight edge energy at ``position`` according to the rule of thirds.

    The weight follows the normalized distance from the frame center, so it
    is continuous and independent of the frame's pixel resolution.  Energy at
    the frame center contributes the neutral weight ``1.0``; the two third
    lines lie at half the maximum distance and receive the peak weight
    ``2.0``; beyond them the weight eases off to ``0.5`` at the frame borders
    (energy located on a third line is still weighted four times as strongly
    as energy at the center).
    """
    normalized_distance = abs(position - 0.5) / (1.0 / 6.0)
    if normalized_distance <= 1.0:
        # Center -> third line: 1.0 up to the peak of 2.0.
        return 1.0 + normalized_distance
    # Third line -> border: 2.0 easing back down to 0.5.
    return 2.0 - 1.5 * (normalized_distance - 1.0)


def composition(frame: Sequence[Sequence[float]]) -> float:
    """Return a normalized rule-of-thirds composition score for ``frame``.

    Both forward-difference orientations are scored: horizontal differences
    (vertical edges) are rewarded on the *vertical* third lines and vertical
    differences (horizontal edges) on the *horizontal* third lines.  The
    weighted energy is divided by the total energy and then by the peak line
    weight (``2.0``), so the result stays in ``[0.0, 1.0]`` and an edge-free
    frame scores ``0.0``.
    """
    grid = _validate_frame(frame)
    rows = len(grid)
    cols = len(grid[0])

    total_energy = 0.0
    weighted_energy = 0.0

    # Horizontal forward differences: vertical edge energy, weighted by the
    # vertical third lines x = 1/3 and x = 2/3.
    if cols >= 2:
        for i in range(rows):
            row = grid[i]
            for j in range(cols - 1):
                energy = abs(row[j + 1] - row[j])
                if energy <= 0.0:
                    continue
                position = (j + 1.0) / cols
                total_energy += energy
                weighted_energy += energy * _line_weight(position)

    # Vertical forward differences: horizontal edge energy, weighted by the
    # horizontal third lines y = 1/3 and y = 2/3.
    if rows >= 2:
        for i in range(rows - 1):
            upper = grid[i]
            lower = grid[i + 1]
            for j in range(cols):
                energy = abs(lower[j] - upper[j])
                if energy <= 0.0:
                    continue
                position = (i + 1.0) / rows
                total_energy += energy
                weighted_energy += energy * _line_weight(position)

    if total_energy <= 0.0:
        return 0.0

    # The peak line weight is 2.0, so weighted_energy / total_energy lies in
    # [0.5, 2.0]; dividing by the peak maps the score onto [0.25, 1.0] while
    # keeping the relative ordering between frames.
    return _clamp_unit(weighted_energy / total_energy / 2.0)


# ---------------------------------------------------------------------------
# Aggregation and selection
# ---------------------------------------------------------------------------


def total_score(
    frame: Sequence[Sequence[float]],
    weights: Mapping[str, float],
) -> float:
    """Return the normalized weighted total score for ``frame``.

    Every weight is non-negative and the weighted sum is divided by the total
    weight, so unnormalized weights cannot overflow the ``[0.0, 1.0]`` range
    and scaling all weights by a common factor leaves the score unchanged.
    """
    grid = _validate_frame(frame)
    validated_weights = _validate_weights(weights)

    components = {
        "sharpness": sharpness(grid),
        "brightness": brightness_score(grid, DEFAULT_BRIGHTNESS_TARGET),
        "contrast": contrast_score(grid),
        "face": face_proxy(grid, (0.25, 0.25, 0.75, 0.75)),
        "composition": composition(grid),
    }

    weight_sum = sum(validated_weights.values())
    weighted_sum = sum(
        validated_weights[key] * components[key] for key in _WEIGHT_KEYS
    )
    return _clamp_unit(weighted_sum / weight_sum)


def _validate_candidates(candidates: Any) -> Sequence[Mapping[str, Any]]:
    """Validate the common candidate sequence shape."""
    if isinstance(candidates, (str, bytes, Mapping)) or not isinstance(
        candidates, Sequence
    ):
        raise CoverError("candidates must be a non-empty sequence")
    if len(candidates) == 0:
        raise CoverError("candidate list must not be empty")

    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise CoverError("each candidate must be a mapping")
        if "idx" not in candidate or "frame" not in candidate:
            raise CoverError(
                "each candidate needs an 'idx' and a 'frame'"
            )
        idx = candidate["idx"]
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise CoverError("candidate 'idx' must be an integer")
        # Reject malformed frames eagerly, including inside batch validation.
        _validate_frame(candidate["frame"])

    return candidates


def _select(candidates: Any, weights: Mapping[str, float]) -> int:
    """Pick the highest-scoring candidate, keeping the earliest one on ties."""
    validated = _validate_candidates(candidates)

    best_idx: int | None = None
    best_score = -math.inf
    for candidate in validated:
        score = total_score(candidate["frame"], weights)
        # A later candidate must only replace the incumbent on a strict,
        # tolerance-aware improvement, so equal scores keep the first winner.
        if best_idx is None or score > best_score + _TOL:
            best_idx = candidate["idx"]
            best_score = score
    return int(best_idx)


def pick_best(candidates: Sequence[Mapping[str, Any]]) -> int:
    """Return the ``idx`` of the highest-scoring cover candidate."""
    return _select(candidates, DEFAULT_WEIGHTS)


def cover_batch(
    videos: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, int]:
    """Select one cover candidate for every video in ``videos``."""
    if isinstance(videos, (str, bytes, Mapping)) or not isinstance(
        videos, Sequence
    ):
        raise CoverError("videos must be a sequence")
    if not isinstance(config, Mapping):
        raise CoverError("config must be a mapping")

    allowed_config_keys = {"weights"}
    for key in config:
        if key not in allowed_config_keys:
            raise CoverError(f"unknown config key: {key!r}")

    weights = config.get("weights", DEFAULT_WEIGHTS)
    # Validated up front so a zero-sum mapping fails before any video is
    # processed and no partial result is ever produced.
    validated_weights = _validate_weights(weights)

    seen_ids: set[Any] = set()
    for video in videos:
        if not isinstance(video, Mapping):
            raise CoverError("each video must be a mapping")
        if "id" not in video or "candidates" not in video:
            raise CoverError("each video needs an 'id' and 'candidates'")
        video_id = video["id"]
        try:
            hash(video_id)
        except TypeError as exc:
            raise CoverError("video 'id' must be hashable") from exc
        if video_id in seen_ids:
            raise CoverError(f"duplicate video id: {video_id!r}")
        seen_ids.add(video_id)
        # Validates the non-empty list and every frame before selection.
        _validate_candidates(video["candidates"])

    # Each video gets a fresh selection pass; no score state is shared between
    # videos, so rankings cannot leak across the batch.
    result: dict[Any, int] = {}
    for video in videos:
        result[video["id"]] = _select(video["candidates"], validated_weights)
    return result
