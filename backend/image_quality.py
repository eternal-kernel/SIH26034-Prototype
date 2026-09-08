"""Image Quality Gate for the SIH26034 prototype (pre-OCR).

Purpose: cheaply flag obviously unusable product photos BEFORE OCR/extraction,
so the inspector can be asked to retake the photo instead of OCR silently
reading garbage.

Deterministic. OpenCV + NumPy only. No LLM, no network.

VERY IMPORTANT — SCOPE:
  This module reports *capture quality* only. It NEVER makes a legal or
  compliance decision. A poor-quality image is a photography problem, not a
  Legal Metrology violation. There is deliberately no "compliant"/"violation"
  field anywhere in this module's output, and downstream code must not treat a
  quality issue as a finding.

Core principle: AI/OCR READS. RULES EVALUATE. EVIDENCE SUPPORTS. INSPECTOR DECIDES.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# Fixed issue-code vocabulary (kept small and stable for the frontend).
UNDECODABLE = "UNDECODABLE"
TOO_SMALL = "TOO_SMALL"
BLURRY = "BLURRY"
TOO_DARK = "TOO_DARK"
TOO_BRIGHT = "TOO_BRIGHT"
LOW_CONTRAST = "LOW_CONTRAST"

# Issues that make an image unusable for OCR (hard blockers).
_HARD_BLOCKERS = {UNDECODABLE, TOO_SMALL}


@dataclass
class QualityThresholds:
    """Prototype capture-quality thresholds.

    THESE ARE NOT LEGAL METROLOGY STANDARDS. They are heuristic, camera- and
    lighting-dependent knobs for deciding whether OCR can plausibly read an
    image. They should be calibrated against real sample photos and tuned per
    deployment. Nothing here defines what a package must declare or whether a
    declaration is lawful.
    """

    # Minimum image side (px). Below this there is likely too little detail.
    min_side: int = 300

    # Blur: variance of the Laplacian on the size-normalized grayscale image.
    # Below blur_min -> flagged BLURRY. Below blur_block -> severe (blocker).
    blur_min: float = 100.0
    blur_block: float = 30.0

    # Brightness (mean grayscale, 0-255).
    dark_min: float = 40.0
    bright_max: float = 220.0

    # Contrast (std of grayscale).
    contrast_min: float = 15.0

    # Clipping: pixels <= clip_dark_level are "crushed"; >= clip_bright_level
    # are "blown". If more than clip_max of the image is clipped either way,
    # the image is treated as too dark / too bright respectively.
    clip_dark_level: int = 15
    clip_bright_level: int = 240
    clip_max: float = 0.5

    # Longest side (px) the image is scaled to before the blur metric, so a
    # given blur_min means roughly the same across resolutions.
    norm_long_side: int = 1000

    # Policy knob: treat severe blur as a hard blocker.
    block_on_severe_blur: bool = True


@dataclass
class ImageQualityResult:
    """Structured, frontend-friendly capture-quality result.

    ``usable`` answers only "can OCR plausibly run on this image?" — never
    "is the product compliant?". ``metrics`` and ``thresholds`` are preserved
    verbatim so a UI can explain exactly why an image needs review.
    """

    usable: bool
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    thresholds: Dict[str, Any] = field(default_factory=dict)
    source_image: Optional[str] = None
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _to_grayscale_u8(image: np.ndarray) -> np.ndarray:
    """Return a uint8 single-channel view of ``image`` (BGR/BGRA/gray in)."""
    if image.ndim == 2:
        gray = image
    else:
        channels = image.shape[2]
        if channels == 1:
            gray = image[:, :, 0]
        elif channels == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = image[:, :, 0]
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def _normalized_for_blur(gray: np.ndarray, long_side: int) -> np.ndarray:
    """Downscale (never upscale) so the longest side ~= ``long_side`` px."""
    h, w = gray.shape[:2]
    longest = max(h, w)
    if longest > long_side and longest > 0:
        scale = long_side / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return gray


def analyze_image(
    image: Optional[np.ndarray],
    thresholds: Optional[QualityThresholds] = None,
    source_image: Optional[str] = None,
) -> ImageQualityResult:
    """Analyze an already-decoded image (NumPy/OpenCV BGR, BGRA, or grayscale).

    This is the primary API. Returns an :class:`ImageQualityResult`. Invalid or
    empty input yields an UNDECODABLE, unusable result rather than raising.
    """
    t = thresholds or QualityThresholds()
    thresholds_dict = asdict(t)

    # --- Decodability / validity ---------------------------------------
    if (
        image is None
        or not isinstance(image, np.ndarray)
        or image.size == 0
        or image.ndim not in (2, 3)
    ):
        return ImageQualityResult(
            usable=False,
            issues=[UNDECODABLE],
            metrics={},
            thresholds=thresholds_dict,
            source_image=source_image,
            note="image could not be decoded or is not a valid 2D/3D array",
        )

    h, w = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]

    gray = _to_grayscale_u8(image)
    norm = _normalized_for_blur(gray, t.norm_long_side)

    # --- Signals --------------------------------------------------------
    laplacian_var = float(cv2.Laplacian(norm, cv2.CV_64F).var())
    brightness_mean = float(gray.mean())
    contrast_std = float(gray.std())
    total = int(gray.size)
    dark_fraction = float(np.count_nonzero(gray <= t.clip_dark_level) / total)
    bright_fraction = float(np.count_nonzero(gray >= t.clip_bright_level) / total)

    metrics = {
        "width": int(w),
        "height": int(h),
        "channels": int(channels),
        "laplacian_var": round(laplacian_var, 3),
        "brightness_mean": round(brightness_mean, 3),
        "contrast_std": round(contrast_std, 3),
        "dark_fraction": round(dark_fraction, 5),
        "bright_fraction": round(bright_fraction, 5),
    }

    # --- Issue detection ------------------------------------------------
    issues: List[str] = []

    if min(h, w) < t.min_side:
        issues.append(TOO_SMALL)

    severe_blur = laplacian_var < t.blur_block
    if laplacian_var < t.blur_min:
        issues.append(BLURRY)

    if brightness_mean < t.dark_min or dark_fraction > t.clip_max:
        issues.append(TOO_DARK)

    if brightness_mean > t.bright_max or bright_fraction > t.clip_max:
        issues.append(TOO_BRIGHT)

    if contrast_std < t.contrast_min:
        issues.append(LOW_CONTRAST)

    # --- Usable policy (capture quality only; NEVER a compliance verdict) -
    usable = True
    if any(code in _HARD_BLOCKERS for code in issues):
        usable = False
    if t.block_on_severe_blur and BLURRY in issues and severe_blur:
        usable = False

    if usable and not issues:
        note = "no obvious capture-quality problems detected"
    elif usable:
        note = "advisory capture-quality issue(s) detected; OCR may still run: " + ", ".join(issues)
    else:
        note = "image is likely unusable for OCR (retake suggested): " + ", ".join(issues)

    return ImageQualityResult(
        usable=usable,
        issues=issues,
        metrics=metrics,
        thresholds=thresholds_dict,
        source_image=source_image,
        note=note,
    )


def analyze_image_path(
    path: str,
    thresholds: Optional[QualityThresholds] = None,
) -> ImageQualityResult:
    """Convenience wrapper: decode ``path`` with OpenCV then analyze it."""
    t = thresholds or QualityThresholds()
    if not isinstance(path, str) or not os.path.isfile(path):
        return ImageQualityResult(
            usable=False,
            issues=[UNDECODABLE],
            metrics={},
            thresholds=asdict(t),
            source_image=path if isinstance(path, str) else None,
            note="file does not exist or is not a regular file",
        )
    image = cv2.imread(path)
    if image is None:
        return ImageQualityResult(
            usable=False,
            issues=[UNDECODABLE],
            metrics={},
            thresholds=asdict(t),
            source_image=path,
            note="OpenCV could not decode the image (unsupported/corrupt?)",
        )
    return analyze_image(image, t, source_image=path)
