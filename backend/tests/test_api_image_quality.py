"""API tests for the POST /image-quality endpoint (Image Quality Gate).

Uses FastAPI's TestClient (httpx-based). Images are synthesized in-memory and
encoded to PNG bytes — no committed fixtures. The endpoint accepts the raw
image bytes as the request body and returns the ImageQualityResult JSON.

This test covers ONLY the quality gate. OCR and extraction are not involved.

Runnable directly (``python tests/test_api_image_quality.py`` from backend/) or
via pytest.
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
from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402

client = TestClient(app)
_ENDPOINT = "/image-quality"


def _png_bytes(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    assert ok, "failed to encode synthetic PNG"
    return buf.tobytes()


def _noise_png(h=600, w=800, seed=1234) -> bytes:
    rng = np.random.default_rng(seed)
    return _png_bytes(rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8))


def test_valid_image_is_usable():
    resp = client.post(
        _ENDPOINT,
        content=_noise_png(),
        params={"filename": "good.png"},
        headers={"Content-Type": "image/png"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["usable"] is True, body
    assert body["issues"] == [], body
    assert body["source_image"] == "good.png"       # filename preserved
    assert "metrics" in body and "thresholds" in body
    return body


def test_too_small_image_is_blocked():
    resp = client.post(_ENDPOINT, content=_noise_png(h=50, w=50))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "TOO_SMALL" in body["issues"], body
    assert body["usable"] is False
    return body


def test_invalid_image_bytes_are_undecodable():
    resp = client.post(_ENDPOINT, content=b"this is not an image")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["issues"] == ["UNDECODABLE"], body
    assert body["usable"] is False
    return body


def test_empty_body_is_undecodable():
    resp = client.post(_ENDPOINT, content=b"")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["issues"] == ["UNDECODABLE"], body
    assert body["usable"] is False
    return body


def test_health_still_works():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    return resp.json()


if __name__ == "__main__":
    checks = [
        ("valid image -> usable", test_valid_image_is_usable),
        ("too-small image -> blocked", test_too_small_image_is_blocked),
        ("invalid bytes -> UNDECODABLE", test_invalid_image_bytes_are_undecodable),
        ("empty body -> UNDECODABLE", test_empty_body_is_undecodable),
        ("health still works", test_health_still_works),
    ]
    for name, fn in checks:
        body = fn()
        print(f"PASS: {name}")
        print("  ->", body)
    print("\nAll API image-quality tests passed.")
