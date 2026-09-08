"""Deterministic, rule-based field extraction from PaddleOCR detections.

This package turns raw OCR detections into structured *candidates*. It never
makes legal / compliance decisions and never invents data: uncertain reads are
flagged as uncertain rather than corrected.

Core principle: AI READS. RULES EVALUATE. EVIDENCE SUPPORTS. INSPECTOR DECIDES.
"""
