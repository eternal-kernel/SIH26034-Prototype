"""SIH26034 — Legal Metrology inspection-assistance backend.

Entry point for the FastAPI app. Currently exposes:
  - GET  /health         : liveness check
  - POST /image-quality  : pre-OCR Image Quality Gate over one uploaded image

OCR (PaddleOCR), the extraction modules, and the rules engine are intentionally
NOT wired in yet. This file must not duplicate any analysis logic — the quality
check delegates entirely to image_quality.analyze_image().

Core principle: AI READS. RULES EVALUATE. EVIDENCE SUPPORTS. INSPECTOR DECIDES.
"""

from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, Request

from image_quality import analyze_image

app = FastAPI(
    title="SIH26034 Backend",
    description="Legal Metrology packaged-commodity inspection-assistance API (skeleton).",
    version="0.1.0",
)


@app.get("/health")
def health():
    """Report that the backend service is running."""
    return {"status": "ok", "service": "SIH26034 backend"}


def _decode_image(raw: bytes):
    """Decode raw image bytes into an OpenCV BGR image, or None if undecodable.

    Returning None (rather than raising) lets analyze_image() produce the
    standard UNDECODABLE quality result for corrupt/empty input.
    """
    if not raw:
        return None
    buffer = np.frombuffer(raw, dtype=np.uint8)
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)  # None on failure


@app.post("/image-quality")
async def image_quality(request: Request, filename: Optional[str] = None):
    """Run the pre-OCR Image Quality Gate over one uploaded image.

    Request body: the raw image bytes (e.g. Content-Type image/png, image/jpeg,
    or application/octet-stream). Optional ``?filename=`` is preserved as
    ``source_image``.

    Returns the ImageQualityResult as JSON. This is a capture-quality signal
    only — it never makes a legal/compliance decision, and it does NOT run OCR
    or extraction. Invalid/corrupt data yields the UNDECODABLE result, not an
    error.
    """
    raw = await request.body()
    image = _decode_image(raw)
    result = analyze_image(image, source_image=filename)
    return result.to_dict()
