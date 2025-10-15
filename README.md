kraft — FFmpeg command generator (Gemini)
========================================

This is a minimal FastAPI backend that turns natural language into a single, ready-to-run FFmpeg command using Google's Gemini API (gemini-2.5-pro).

Features
- Upload multiple files (video/audio/image). Their filenames are passed as context to the LLM.
- Default output directory configured in `config.json` (created on first run). Outputs must be saved there.
- Returns a single FFmpeg command as plain text. No explanations.

Requirements
- Python 3.12+
- A Google Gemini API key in your environment: `GEMINI_API_KEY`.

Setup
1) Install deps
	Use your preferred tool (pip, uv, etc.). Example with pip:

	pip install -e .

	Or directly from pyproject:

	pip install fastapi uvicorn google-genai python-dotenv python-multipart

2) Create `.env`
	Copy `.env.example` to `.env` and set `GEMINI_API_KEY`.

3) Configure output path
	Edit `config.json` if needed. Default is `outputs/`.

Run the API
	uvicorn main:app --reload

Endpoints
- GET /health → "ok"
- POST /generate (multipart/form-data)
  Fields:
  - prompt: string (required) — your natural language description
  - files: one or more file parts (optional) — the media inputs

Response: text/plain body containing only the FFmpeg command.

Notes
- Uploaded files are saved under `uploads/` with their original names. The LLM is instructed to reference inputs by filename only (same working dir). If you run the returned command from the project root, inputs should resolve correctly if you copy or run from `uploads/`. Adjust as needed in your integration.
- The backend sanitizes output: rejects shell metacharacters like `;`, `&&`, `|`, `>` and ensures a single line.
- If the model omits the output path for the final output token, the backend will prepend the configured `default_output_path`.

License
MIT
