"""Unit tests for image_quality (synthetic, in-memory images).

Images are generated deterministically with NumPy/OpenCV — no committed image
fixtures (product images are git-ignored). One optional test uses the real
amul_icecream.jpeg if it is present locally, and skips otherwise.

Runnable directly (``python tests/test_image_quality.py`` from backend/) or via
pytest.
"""

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from image_quality import (  # noqa: E402
    QualityThresholds,
    analyze_image,
    analyze_image_path,
)

_REPO_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AMUL_PATH = os.path.join(_REPO_BACKEND, "amul_icecream.jpeg")


def _noise_image(h=600, w=800, seed=1234):
    """A sharp, well-exposed, high-contrast image (uniform random noise)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _solid(h, w, value):
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_good_image_has_no_issues():
    result = analyze_image(_noise_image(), source_image="good.png")
    assert result.usable is True, result.note
    assert result.issues == [], result.issues
    return result


def test_blurred_image_flags_blurry():
    blurred = cv2.GaussianBlur(_noise_image(), (31, 31), 0)
    result = analyze_image(blurred, source_image="blurred.png")
    assert "BLURRY" in result.issues, result.metrics
    return result


def test_too_dark_image():
    result = analyze_image(_solid(600, 800, 10), source_image="dark.png")
    assert "TOO_DARK" in result.issues, result.metrics
    return result


def test_too_bright_image():
    result = analyze_image(_solid(600, 800, 250), source_image="bright.png")
    assert "TOO_BRIGHT" in result.issues, result.metrics
    return result


def test_low_contrast_image():
    result = analyze_image(_solid(600, 800, 128), source_image="flat.png")
    assert "LOW_CONTRAST" in result.issues, result.metrics
    return result


def test_too_small_image_is_hard_blocker():
    result = analyze_image(_noise_image(h=50, w=50), source_image="tiny.png")
    assert "TOO_SMALL" in result.issues
    assert result.usable is False, "TOO_SMALL must be a hard blocker"
    return result


def test_undecodable_input():
    # None input
    r_none = analyze_image(None, source_image="none")
    assert r_none.issues == ["UNDECODABLE"]
    assert r_none.usable is False
    # Missing file path
    r_missing = analyze_image_path(os.path.join(_REPO_BACKEND, "does_not_exist.png"))
    assert r_missing.issues == ["UNDECODABLE"]
    assert r_missing.usable is False
    return r_none


def test_threshold_and_metrics_transparency():
    result = analyze_image(_noise_image(), source_image="good.png")
    # Metrics expose the exact signals the frontend can display.
    for key in (
        "width", "height", "channels", "laplacian_var", "brightness_mean",
        "contrast_std", "dark_fraction", "bright_fraction",
    ):
        assert key in result.metrics, key
    # Default thresholds are echoed back verbatim.
    from dataclasses import asdict
    assert result.thresholds == asdict(QualityThresholds())
    # Custom thresholds are honored and reflected in the result.
    custom = QualityThresholds(min_side=10_000)  # force TOO_SMALL on any test image
    forced = analyze_image(_noise_image(), thresholds=custom)
    assert "TOO_SMALL" in forced.issues
    assert forced.thresholds["min_side"] == 10_000
    return result


def test_real_amul_image_if_available():
    """Optional: the real Amul photo should pass the gate. Skips if absent."""
    if not os.path.isfile(_AMUL_PATH):
        print("SKIP: real Amul image not present at", _AMUL_PATH)
        return None
    result = analyze_image_path(_AMUL_PATH)
    print("  real Amul quality:", result.to_dict())
    assert result.usable is True, f"real Amul image unexpectedly unusable: {result.note}"
    return result


if __name__ == "__main__":
    checks = [
        ("good image -> no issues", test_good_image_has_no_issues),
        ("blurred -> BLURRY", test_blurred_image_flags_blurry),
        ("too dark -> TOO_DARK", test_too_dark_image),
        ("too bright -> TOO_BRIGHT", test_too_bright_image),
        ("low contrast -> LOW_CONTRAST", test_low_contrast_image),
        ("too small -> hard blocker", test_too_small_image_is_hard_blocker),
        ("undecodable input", test_undecodable_input),
        ("threshold/metrics transparency", test_threshold_and_metrics_transparency),
    ]
    for name, fn in checks:
        result = fn()
        print(f"PASS: {name}")
        if result is not None:
            print("  ->", {"usable": result.usable, "issues": result.issues, "metrics": result.metrics})

    print()
    real = test_real_amul_image_if_available()
    print("PASS: real Amul image (or skipped)")

    print("\nAll image-quality tests passed.")
