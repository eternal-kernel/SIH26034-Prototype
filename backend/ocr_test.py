"""Standalone PaddleOCR test script for the SIH26034 prototype.

Purpose: run PaddleOCR (3.7.0 API) on a single local image and print the raw
detections, recognized text, confidence values, and bounding-box coordinates.

This is a diagnostic tool ONLY. It does NOT:
  - expose a FastAPI endpoint,
  - extract structured fields (Manufacturer, Net Quantity, MRP, etc.),
  - apply any Legal Metrology rules,
  - decide whether anything is compliant.

Usage:
    python ocr_test.py <path-to-image>

Core principle: AI READS. RULES EVALUATE. EVIDENCE SUPPORTS. INSPECTOR DECIDES.
"""

import os
import sys

# PaddleOCR can recognize characters (e.g. arrows, currency, non-Latin) that
# the default Windows console codepage (cp1252) cannot encode. Force UTF-8 so
# printing detections never crashes on such characters.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001 - older/odd streams may not support reconfigure
    pass


def main(argv):
    # --- Argument / path validation -------------------------------------
    if len(argv) != 2:
        print("Usage: python ocr_test.py <path-to-image>")
        return 2

    image_path = argv[1]

    if not os.path.exists(image_path):
        print(f"ERROR: file does not exist: {image_path}")
        return 1
    if not os.path.isfile(image_path):
        print(f"ERROR: path is not a file: {image_path}")
        return 1

    # --- Load the image (validates it is a readable image) --------------
    try:
        import cv2
    except ImportError:
        print("ERROR: OpenCV (cv2) is not available in this environment.")
        return 1

    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: could not decode image (unsupported/corrupt?): {image_path}")
        return 1
    print(f"Loaded image: {image_path}  (shape={image.shape})")

    # --- Run PaddleOCR (3.7.0 predict API) ------------------------------
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        print(f"ERROR: could not import PaddleOCR: {exc}")
        return 1

    print("Initializing PaddleOCR (first run may download models)...")
    # enable_mkldnn=False avoids a PaddlePaddle 3.x oneDNN/PIR inference crash
    # ("ConvertPirAttribute2RuntimeAttribute not support ...") on CPU.
    # The document-preprocessing models are disabled: they are unnecessary for
    # flat label photos and reduce the failure surface / startup cost.
    ocr = PaddleOCR(
        lang="en",
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    try:
        results = ocr.predict(image)
    except Exception as exc:  # noqa: BLE001 - surface any OCR failure clearly
        print(f"ERROR: OCR prediction failed: {exc}")
        return 1

    if not results:
        print("No results returned by OCR.")
        return 0

    # --- Report ----------------------------------------------------------
    for page_index, res in enumerate(results):
        print("\n" + "=" * 60)
        print(f"RESULT PAGE {page_index}")
        print("=" * 60)

        # 1) Raw detections, exactly as PaddleOCR structures them.
        print("\n--- RAW OCR RESULT ---")
        try:
            res.print()  # PaddleOCR's own formatted dump of the full result
        except Exception:  # noqa: BLE001
            print(res)

        # Dict-like access to the individual fields (PaddleOCR 3.x keys).
        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])
        # Prefer recognition polygons; fall back to axis-aligned boxes or
        # raw detection polygons depending on what this version provides.
        boxes = res.get("rec_polys")
        if boxes is None:
            boxes = res.get("rec_boxes")
        if boxes is None:
            boxes = res.get("dt_polys")
        if boxes is None:
            boxes = []

        # 2) Detected text, 3) confidence, 4) bounding boxes.
        print("\n--- DETECTED TEXT / CONFIDENCE / BOUNDING BOX ---")
        print(f"Total detections: {len(texts)}")
        for i, text in enumerate(texts):
            score = scores[i] if i < len(scores) else None
            box = boxes[i] if i < len(boxes) else None

            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)

            # Boxes are numpy arrays; convert to plain lists for readable output.
            if box is not None and hasattr(box, "tolist"):
                box = box.tolist()

            print(f"[{i}] text={text!r}  confidence={score_str}  box={box}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
