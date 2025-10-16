# ⚡️ kraft — FFmpeg Command Summoner

> ✨ "Type a vibe, get a battle-tested `ffmpeg` spell." — kraft mission control

kraft turns natural-language prompts into production-ready `ffmpeg` commands with a Gemini-powered generator, an always-on safety critic, and a playful neon UI.

## 🌟 Why You'll Love It
- 🧠 **Prompt → Command pipeline** with iterative generator/critic loop for reliable output
- 🎛️ **Retro-inspired web UI** in `static/index.html` for chatty command crafting
- 🛡️ **Safety rails** that ban shell escapes, keep outputs in `outputs/`, and auto-sanitize paths
- 📝 **Gemini explains** the final command so you know exactly what will run
- ⚙️ **`/execute` endpoint** to run the command server-side with timeout + output capture

## 🚀 Quickstart
```bash
# 1. Install deps with uv (preferred)
uv sync  # creates .venv and installs dependencies from pyproject.toml

# (optional) Activate the environment if you need direct access
source .venv/bin/activate

# (optional) Using plain pip instead:
python -m venv .venv
source .venv/bin/activate
pip install -e .

# 2. Configure environment
printf "GEMINI_API_KEY=your_api_key_here\n" > .env    # or export in your shell profile

# 3. Launch the backend
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 4. Visit the UI
xdg-open http://localhost:8000  # use open/start on macOS/Windows
```

> 💡 **Tip:** The service refuses to run without `GEMINI_API_KEY`, so load it via `.env` or your shell.

## 🗂️ Project Atlas
```text
.
├── main.py              # FastAPI app: generator loop, critic, execution endpoint
├── config.json          # Default output directory override
├── static/              # Retro UI (Tailwind + dialog system)
├── uploads/             # Temp home for user media uploads
├── outputs/             # Auto-created output drop zone
└── pyproject.toml       # Poetry-core metadata & dependencies
```

## ⚙️ Configuration Cheatsheet
- `config.json`
  - `default_output_path`: relative folder where all finished media lands (default `outputs`).
- Environment variables
  - `GEMINI_API_KEY` *(required)*: Google Generative AI key for command + explanation models.
  - Optional FastAPI knobs: set `UVICORN_RELOAD`, `PORT`, etc., via your process manager.

## 🧪 How Command Generation Works
1. 📨 Upload media + describe your dream transformation.
2. 🧮 `build_instruction()` crafts a strict system prompt with allowed inputs & rules.
3. 🤖 `call_gemini()` proposes a single-line `ffmpeg` command.
4. 🧼 `extract_and_sanitize_command()` rewrites paths, forbids sneaky tokens, and normalizes output.
5. 🕵️ `call_gemini_critic()` double-checks compliance and can suggest a patched command.
6. 💬 (Optional) `call_gemini_explain()` narrates what the command will do.
7. 🏁 `/execute` endpoint can run the command with `-y` safety, 600s timeout, and output capture.

## 🕹️ Frontend Highlights (`static/index.html`)
- 💬 Animated quote box + dialog stack keeps the vibe alive.
- 🎨 Effect Blocks auto-fill prompts for common edits.
- ⚡ Copy-and-run controls with confirmation overlays.
- 🧭 Live reviews panel shows every critic iteration and timing metrics returned from the API.

## 📡 API Endpoints
- `GET /health` → quick uptime check (`ok`).
- `GET /` → serves the Tailwind UI from `static/index.html`.
- `POST /generate` *(multipart form)*
  - `prompt`: natural language request.
  - `files[]`: optional media uploads.
  - `enable_critic`, `max_iters`, `enable_explain`, `output_path`: advanced knobs.
  - Response returns the final command, explanation, critic reviews, timing, and file metadata.
- `POST /execute`
  - Runs a validated command server-side.
  - Rejects anything that does not start with `ffmpeg` or uses forbidden tokens.

## 🧭 Dev Workflow
1. Hack on `main.py` for backend logic.
2. Tweak the UI under `static/` with Tailwind + vanilla JS.
3. Use the FastAPI docs at `http://localhost:8000/docs` (custom theme + favicon included).
4. Keep uploads small—everything goes to `uploads/` and is hashed/ffprobed automatically.

## 🧱 Roadmap Ideas
- 🤹 Drag-and-drop timeline for chaining multiple prompts.
- 🔍 Inline ffprobe previews inside the UI.
- 🧠 Model selection dropdown (flash vs. pro) with token budgeting hints.
- 📼 Preset gallery for platform-specific exports (TikTok, YouTube Shorts, podcasts).

## 🤝 Contributing
- Fork it, branch it, craft your upgrade.
- Add or update tests if you wire new flows.
- Keep comments tidy; the code already narrates the tricky parts.
- Open a PR with screenshots/GIFs—retro vibes deserve to be seen.

Made with ☕, 🎧, and a dash of `ffmpeg` wizardry.
