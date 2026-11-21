"""Automatic video cover selection package."""

from .scoring import (
    DEFAULT_BRIGHTNESS_TARGET,
    DEFAULT_WEIGHTS,
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
