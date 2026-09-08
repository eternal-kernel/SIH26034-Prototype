# SIH26034 Backend

Minimal FastAPI backend skeleton for the Legal Metrology packaged-commodity
inspection-assistance prototype.

At this stage the backend only exposes a health-check endpoint. OCR, OpenCV,
the rules engine, the database, and frontend integration are **not** part of
this skeleton and will be added in later tasks.

## Requirements

- Python 3.10+ (developed against Python 3.13)

## Setup

From the `backend/` directory:

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# Windows (cmd):
venv\Scripts\activate.bat
# macOS / Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Run the server

From the `backend/` directory:

```bash
uvicorn main:app --reload --port 8000
```

The server starts on http://127.0.0.1:8000

- Health check: http://127.0.0.1:8000/health
- Interactive API docs: http://127.0.0.1:8000/docs

## Health endpoint

```
GET /health
```

Response:

```json
{ "status": "ok", "service": "SIH26034 backend" }
```
