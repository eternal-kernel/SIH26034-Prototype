"""SIH26034 — Legal Metrology inspection-assistance backend.

Minimal FastAPI skeleton. This file only provides the application entry point
and a health-check endpoint. OCR, OpenCV, the rules engine, the database, and
frontend integration are intentionally NOT implemented here.

Core principle: AI READS. RULES EVALUATE. EVIDENCE SUPPORTS. INSPECTOR DECIDES.
"""

from fastapi import FastAPI

app = FastAPI(
    title="SIH26034 Backend",
    description="Legal Metrology packaged-commodity inspection-assistance API (skeleton).",
    version="0.1.0",
)


@app.get("/health")
def health():
    """Report that the backend service is running."""
    return {"status": "ok", "service": "SIH26034 backend"}
