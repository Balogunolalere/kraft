import os
import json
import re
import shlex
from pathlib import Path
from typing import List, Optional, Any, Tuple

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import subprocess
import hashlib
import datetime
import time
from dotenv import load_dotenv

from google import genai
from google.genai import types


# --- Startup: env, config, dirs ---
load_dotenv()

ROOT = Path(__file__).parent.resolve()
CONFIG_PATH = ROOT / "config.json"
DEFAULT_CONFIG = {"default_output_path": "outputs"}


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2))
    try:
        cfg = json.loads(CONFIG_PATH.read_text())
        if "default_output_path" not in cfg or not isinstance(cfg["default_output_path"], str):
            raise ValueError("Invalid config.json: missing default_output_path")
        return cfg
    except Exception as e:
        raise RuntimeError(f"Failed to load config.json: {e}")


config = load_config()
OUTPUT_DIR = ROOT / config["default_output_path"]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UPLOADS_DIR = ROOT / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def resolve_output_override(value: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Validate and resolve a user-provided output path inside OUTPUT_DIR."""
    if value is None:
        return None, None
    cleaned = value.strip()
    if not cleaned:
        return None, None

    cleaned = cleaned.replace("\\", "/")
    candidate = Path(cleaned)
    if candidate.is_absolute():
        raise HTTPException(status_code=400, detail="Output path must be relative to the outputs directory")
    if any(part == ".." for part in candidate.parts):
        raise HTTPException(status_code=400, detail="Output path cannot traverse outside the outputs directory")

    target = (OUTPUT_DIR / candidate).resolve()
    try:
        relative = target.relative_to(OUTPUT_DIR)
    except ValueError:
        raise HTTPException(status_code=400, detail="Output path must stay within the outputs directory")

    target.parent.mkdir(parents=True, exist_ok=True)

    return str(target), str(relative)


# --- App ---
app = FastAPI(
    title="kraft",
    version="0.1.0",
    description="FFmpeg command generator backed by Gemini",
    docs_url="/docs",
    redoc_url=None,
    swagger_ui_parameters={
        "customSiteTitle": "kraft API",
        "customfavIcon": "/static/kraft-favicon.svg",
        "customCssUrl": "/static/docs.css",
        "displayRequestDuration": True,
        "docExpansion": "list",
        "defaultModelsExpandDepth": -1,
        "defaultModelExpandDepth": 1,
        "tryItOutEnabled": True,
        "persistAuthorization": True,
    },
)

# Mount static assets for docs styling
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


@app.get("/health", response_class=PlainTextResponse)
def health() -> str:
    return "ok"


@app.get("/", response_class=PlainTextResponse)
def root():
    """Serve the main HTML interface"""
    index_path = ROOT / "static" / "index.html"
    if index_path.exists():
        from fastapi.responses import HTMLResponse
        return HTMLResponse(content=index_path.read_text(), status_code=200)
    return PlainTextResponse("Frontend not found", status_code=404)


def build_instruction(
    list_of_filenames: List[str],
    default_output_path: str,
    user_prompt: str,
    previous_command: Optional[str] = None,
    critic_feedback: Optional[str] = None,
    desired_output: Optional[str] = None,
) -> str:
    uploads_dir = str(UPLOADS_DIR)
    full_input_paths = ", ".join(str(UPLOADS_DIR / fn) for fn in list_of_filenames) if list_of_filenames else "(none)"
    
    base = (
        "EXPERT FFMPEG COMMAND GENERATOR\n\n"
        "INPUTS (use these EXACT full paths):\n"
        f"{full_input_paths}\n\n"
        f"OUTPUT DIRECTORY (all outputs MUST go here): {default_output_path}/\n\n"
        "ABSOLUTE REQUIREMENTS:\n"
        "1. Return ONLY the ffmpeg command - ONE LINE - NO explanations, markdown, or fences\n"
        "2. Command MUST start with 'ffmpeg'\n"
        "3. Use double quotes (\") for paths/filters with spaces or special chars\n"
        "4. FORBIDDEN: ; && || | > < ` $() * ? wildcards env-vars subshells pipes redirects\n"
        "5. Use ONLY the provided input paths above - NO other files\n"
        f"6. ALL outputs to {default_output_path}/ with explicit filenames (e.g., output.mp4)\n"
        "7. SINGLE COMMAND ONLY - NO multi-pass encoding with && or multiple commands\n\n"
        "ENCODING DEFAULTS (use only when needed):\n"
        "- MP4: -c:v libx264 -pix_fmt yuv420p -movflags +faststart -c:a aac -b:a 192k\n"
        "- Preserve original: -c copy (if container compatible)\n"
        "- Multiple streams: use -map explicitly\n\n"
        "FILE SIZE TARGETING:\n"
        "- For specific file sizes: Use single-pass with -fs (file size limit) OR estimate bitrate\n"
        "- Calculate bitrate: (target_size_MB * 8192) / duration_seconds = bitrate_kbps\n"
        "- Use -b:v for video bitrate, -b:a for audio bitrate\n"
        "- NEVER use two-pass encoding (no -pass 1, -pass 2, or && operators)\n\n"
    )

    if desired_output:
        base += (
            "OUTPUT TARGET:\n"
            f"- Final output file MUST be exactly: {desired_output}\n"
            "- Do not create additional outputs or alternate filenames\n\n"
        )

    base += (
        "STRICT RULES:\n"
        "- Do EXACTLY what user asks - NO extra filters, scaling, trimming, or changes\n"
        "- Preserve resolution/fps/duration unless specifically requested to change\n"
        "- Choose minimal safe defaults for ambiguities\n"
        "- Cross-platform compatible paths\n\n"
        f"USER REQUEST: {user_prompt}\n"
    )
    
    if previous_command or critic_feedback:
        prev = previous_command or "(none)"
        feed = critic_feedback or "(none)"
        base += (
            f"\nPREVIOUS ATTEMPT: {prev}\n"
            f"ERRORS TO FIX: {feed}\n"
            "Return corrected command only.\n"
        )
    return base


def call_gemini(instruction: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)
    model = "gemini-2.5-flash"
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=instruction)],
        )
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=512,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    chunks: List[str] = []
    for chunk in client.models.generate_content_stream(
        model=model, contents=contents, config=generate_content_config
    ):
        if hasattr(chunk, "text") and chunk.text:
            chunks.append(chunk.text)
    return "".join(chunks).strip()


FORBIDDEN_TOKENS = [";", "&&", "||", "|", ">", "<", "`"]


def extract_and_sanitize_command(
    text: str,
    default_output_path: str,
    filenames: List[str],
    uploads_dir: str,
    *,
    forced_output_path: Optional[str] = None,
    check_forbidden: bool = True,
) -> str:
    # Extract first line that starts with 'ffmpeg'
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("```")]
    candidate: Optional[str] = None
    for ln in lines:
        if ln.lower().startswith("ffmpeg ") or ln.lower() == "ffmpeg":
            candidate = ln
            break
    if candidate is None:
        # Fallback: join all text and look for 'ffmpeg'
        m = re.search(r"(?im)^(ffmpeg\b.*)$", text)
        if m:
            candidate = m.group(1).strip()
        else:
            candidate = text.strip()

    # Collapse whitespace/newlines into single spaces
    cmd = " ".join(candidate.split())

    # Remove stray code fences if present
    cmd = cmd.replace("```", "").strip()

    # Validate
    if not cmd.lower().startswith("ffmpeg"):
        raise HTTPException(status_code=502, detail="Model did not return an ffmpeg command")

    if check_forbidden:
        for tok in FORBIDDEN_TOKENS:
            if tok in cmd:
                raise HTTPException(status_code=400, detail=f"Forbidden token in command: {tok}")

    # Ensure single line
    if "\n" in cmd or "\r" in cmd:
        cmd = re.sub(r"\s+", " ", cmd)

    # Rewrite input filenames to full uploads paths where safe using shlex tokenization
    def quote_token(tok: str) -> str:
        # Prefer double quotes when needed
        if re.search(r"\s|,|\(|\)|\[|\]|\{|\}|;|&|\|", tok):
            escaped = tok.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        return tok

    rewritten = False
    try:
        tokens = shlex.split(cmd)
        if len(tokens) >= 1 and tokens[0].lower() == "ffmpeg":
            last_index = len(tokens) - 1
            for i in range(1, last_index):  # avoid touching the last token which is likely output
                t = tokens[i]
                for name in filenames:
                    full = str(Path(uploads_dir) / name)
                    if t == name:
                        tokens[i] = full
                        t = tokens[i]
                        rewritten = True
                    if f"={name}" in t:
                        tokens[i] = t.replace(f"={name}", f'="{full}"')
                        t = tokens[i]
                        rewritten = True
            if forced_output_path:
                if not tokens:
                    tokens = ["ffmpeg", forced_output_path]
                elif tokens[-1].startswith("-"):
                    tokens.append(forced_output_path)
                else:
                    tokens[-1] = forced_output_path
            # Re-join with double-quote preference
            cmd = " ".join(quote_token(tok) for tok in tokens)
    except Exception:
        # Best-effort; on failure, keep original cmd
        pass

    if forced_output_path:
        try:
            tokens = shlex.split(cmd)
        except Exception:
            tokens = cmd.split()
        if not tokens:
            tokens = ["ffmpeg", forced_output_path]
        elif tokens[0].lower() != "ffmpeg":
            tokens = ["ffmpeg"] + tokens
        if tokens[-1].startswith("-"):
            tokens.append(forced_output_path)
        else:
            tokens[-1] = forced_output_path
        cmd = " ".join(quote_token(tok) for tok in tokens)

    if not forced_output_path:
        # Light check that output directory is referenced for outputs (best-effort)
        # If there's a likely output filename at the end without a path, prepend default_output_path
        parts = cmd.split()
        if parts and not parts[-1].startswith("-"):
            last = parts[-1]
            # Heuristic: if last token has an extension and no path separators, ensure it includes output dir
            if ("/" not in last and "\\" not in last) and re.search(r"\.[a-zA-Z0-9]{2,4}$", last):
                parts[-1] = f"{default_output_path}/{last}"
                cmd = " ".join(parts)

    return cmd


def call_gemini_explain(prompt: str, command: str, input_paths: List[str], output_dir: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return ""
    client = genai.Client(api_key=api_key)
    model = "gemini-2.5-flash"
    inputs_str = ", ".join(input_paths) if input_paths else "(none)"
    instruction = (
        f"Explain in 2-4 lines what this ffmpeg command does:\n{command}\n\n"
        f"Inputs: {inputs_str}\nOutput: {output_dir}\n"
    )
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=instruction)])]
    generate_content_config = types.GenerateContentConfig(
        temperature=0.3,
        max_output_tokens=256,
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )
    chunks: List[str] = []
    for chunk in client.models.generate_content_stream(
        model=model, contents=contents, config=generate_content_config
    ):
        if hasattr(chunk, "text") and chunk.text:
            chunks.append(chunk.text)
    return "".join(chunks).strip()


def call_gemini_critic(
    user_prompt: str,
    input_full_paths: List[str],
    default_output_path: str,
    candidate_command: str,
    forbidden_tokens: List[str],
    forced_output: Optional[str] = None,
) -> dict:
    """Ask a strict critic to validate the command; expects JSON response."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"ok": True, "reasons": ["No API key for critic; skipping"], "must_changes": [], "revised_command": None}
    client = genai.Client(api_key=api_key)
    model = "gemini-2.5-flash"
    inputs = ", ".join(input_full_paths) if input_full_paths else "(none)"
    forbid = ", ".join(forbidden_tokens)
    instruction = (
        "STRICT FFMPEG COMMAND VALIDATOR\n\n"
        f"USER REQUEST: {user_prompt}\n"
        f"ALLOWED INPUTS: {inputs}\n"
        f"REQUIRED OUTPUT DIR: {default_output_path}\n"
        f"FORBIDDEN TOKENS: {forbid}\n"
        f"CANDIDATE: {candidate_command}\n\n"
        "VALIDATE AGAINST:\n"
        "1. Uses ONLY allowed input paths (exact match)\n"
        f"2. ALL outputs in {default_output_path}\n"
        "3. NO forbidden tokens (especially && for multi-pass)\n"
        "4. Does EXACTLY what user asked - NO extra operations\n"
        "5. Single-line starting with 'ffmpeg' - ONE command only\n"
        "6. NO two-pass encoding (-pass 1/-pass 2) - use single-pass with bitrate calculation\n\n"
    )

    if forced_output:
        instruction += (
            "OUTPUT CHECK:\n"
            f"- Final command MUST write to exactly this path: {forced_output}\n"
            "- Reject if different output filename/path is used\n\n"
        )

    instruction += (
        'RESPOND WITH ONLY THIS JSON (no fences):\n'
        '{"ok": boolean, "reasons": ["reason1"], "must_changes": ["change1"], "revised_command": "fixed_cmd_or_null"}\n'
    )
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=instruction)])]
    generate_content_config = types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=512,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="application/json"
    )
    chunks: List[str] = []
    for chunk in client.models.generate_content_stream(
        model=model, contents=contents, config=generate_content_config
    ):
        if hasattr(chunk, "text") and chunk.text:
            chunks.append(chunk.text)
    text = "".join(chunks).strip()
    # Try to extract JSON
    cleaned = text.strip().strip("`")
    # Find first '{' to last '}'
    try:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = cleaned[start : end + 1]
            data = json.loads(json_str)
            # Basic normalization
            return {
                "ok": bool(data.get("ok", False)),
                "reasons": list(data.get("reasons", [])),
                "must_changes": list(data.get("must_changes", [])),
                "revised_command": data.get("revised_command"),
            }
    except Exception:
        pass
    # Heuristic fallback
    ok = bool(re.search(r'"ok"\s*:\s*true', cleaned.lower()))
    return {"ok": ok, "reasons": [cleaned[:2000]], "must_changes": [], "revised_command": None}


def probe_file_metadata(path: Path) -> Optional[dict]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-print_format",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)
    except Exception:
        return None


class FileMeta(BaseModel):
    filename: str
    full_path: str
    size_bytes: int
    sha256: Optional[str] = None
    content_type: Optional[str] = None
    ffprobe: Optional[dict] = None


class Review(BaseModel):
    iteration: int
    candidate_command: str
    ok: bool
    reasons: List[str]
    must_changes: List[str]
    revised_command: Optional[str] = None
    gen_s: Optional[float] = None
    critic_s: Optional[float] = None
    sanitize_s: Optional[float] = None


class GenerateResponse(BaseModel):
    prompt: str
    command: str
    explanation: str
    inputs: List[FileMeta]
    default_output_dir: str
    estimated_output_path: Optional[str] = None
    safety: dict
    model: str
    created_at: str
    notes: Optional[List[str]] = None
    reviews: List[Review]
    timings: dict


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    prompt: str = Form(..., description="Natural language description of the desired FFmpeg operation"),
    files: List[UploadFile] = File(default=[]),
    enable_critic: bool = Form(True, description="Run the critic loop to refine commands"),
    max_iters: int = Form(3, description="Max iterations for generator+critic loop"),
    enable_explain: bool = Form(True, description="Ask LLM to explain the final command"),
    output_path: Optional[str] = Form(None, description="Override output path relative to the default outputs directory"),
):
    t_total_start = time.perf_counter()
    t_upload_hash_s = 0.0
    t_ffprobe_s = 0.0
    max_iters = max(1, min(int(max_iters), 6))
    # Save uploaded files (store under uploads/ with original filenames)
    filenames: List[str] = []
    file_metas: List[FileMeta] = []
    for uf in files:
        name = Path(uf.filename).name
        if not name:
            continue
        # Basic filename safety: allow simple names
        if re.search(r"[^A-Za-z0-9._\- ]", name):
            raise HTTPException(status_code=400, detail=f"Unsupported filename: {name}")
        dest = UPLOADS_DIR / name
        content = await uf.read()
        dest.write_bytes(content)
        filenames.append(name)
        t0 = time.perf_counter()
        sha = hashlib.sha256(content).hexdigest()
        t_upload_hash_s += time.perf_counter() - t0
        size = len(content)
        meta = FileMeta(
            filename=name,
            full_path=str(dest),
            size_bytes=size,
            sha256=sha,
            content_type=getattr(uf, "content_type", None),
            ffprobe=None,
        )
        t1 = time.perf_counter()
        meta.ffprobe = probe_file_metadata(dest)
        t_ffprobe_s += time.perf_counter() - t1
        file_metas.append(meta)

    custom_output_abs, custom_output_rel = resolve_output_override(output_path)

    # Iterative generation with critic loop
    reviews: List[Review] = []
    previous_cmd: Optional[str] = None
    critic_feedback_text: Optional[str] = None
    final_cmd: Optional[str] = None
    iter_gen_sums: List[float] = []
    iter_critic_sums: List[float] = []
    iter_sanitize_sums: List[float] = []

    if enable_critic:
        for it in range(1, max_iters + 1):
            instruction = build_instruction(
                filenames,
                str(OUTPUT_DIR),
                prompt,
                previous_command=previous_cmd,
                critic_feedback=critic_feedback_text,
                desired_output=custom_output_abs,
            )
            t_gen0 = time.perf_counter()
            llm_text = call_gemini(instruction)
            gen_s = time.perf_counter() - t_gen0
            t_san0 = time.perf_counter()
            candidate_cmd = extract_and_sanitize_command(
                llm_text,
                str(OUTPUT_DIR),
                filenames,
                str(UPLOADS_DIR),
                forced_output_path=custom_output_abs,
                check_forbidden=False,
            )
            sanitize_s = time.perf_counter() - t_san0

            t_cr0 = time.perf_counter()
            critic = call_gemini_critic(
                prompt,
                [m.full_path for m in file_metas],
                str(OUTPUT_DIR),
                candidate_cmd,
                FORBIDDEN_TOKENS,
                forced_output=custom_output_abs,
            )
            critic_s = time.perf_counter() - t_cr0
            review = Review(
                iteration=it,
                candidate_command=candidate_cmd,
                ok=bool(critic.get("ok", False)),
                reasons=list(critic.get("reasons", [])),
                must_changes=list(critic.get("must_changes", [])),
                revised_command=critic.get("revised_command"),
                gen_s=gen_s,
                critic_s=critic_s,
                sanitize_s=sanitize_s,
            )
            reviews.append(review)
            iter_gen_sums.append(gen_s)
            iter_critic_sums.append(critic_s)
            iter_sanitize_sums.append(sanitize_s)

            if review.ok:
                final_cmd = candidate_cmd
                break
            # If critic provided a revised command, try it next iteration directly
            next_cmd = review.revised_command
            previous_cmd = candidate_cmd
            critic_feedback_text = "; ".join(review.must_changes or review.reasons)
            if next_cmd:
                previous_cmd = next_cmd
    else:
        # Single-shot generation without critic
        instruction = build_instruction(
            filenames,
            str(OUTPUT_DIR),
            prompt,
            desired_output=custom_output_abs,
        )
        t_gen0 = time.perf_counter()
        llm_text = call_gemini(instruction)
        gen_s = time.perf_counter() - t_gen0
        t_san0 = time.perf_counter()
        candidate_cmd = extract_and_sanitize_command(
            llm_text,
            str(OUTPUT_DIR),
            filenames,
            str(UPLOADS_DIR),
            forced_output_path=custom_output_abs,
            check_forbidden=False,
        )
        sanitize_s = time.perf_counter() - t_san0
        reviews.append(
            Review(
                iteration=1,
                candidate_command=candidate_cmd,
                ok=True,
                reasons=["critic disabled"],
                must_changes=[],
                revised_command=None,
                gen_s=gen_s,
                critic_s=0.0,
                sanitize_s=sanitize_s,
            )
        )
        final_cmd = candidate_cmd

    cmd = final_cmd or reviews[-1].candidate_command

    # Estimated output path (best-effort: last token if not a flag)
    est_output = custom_output_abs
    if est_output is None:
        parts = cmd.split()
        if parts and not parts[-1].startswith("-"):
            est_output = parts[-1]

    # Safety report
    forbidden_found = [tok for tok in FORBIDDEN_TOKENS if tok in cmd]
    safety = {"forbidden_tokens_found": forbidden_found}

    # Explanation via LLM (optional)
    t_explain0 = time.perf_counter()
    explanation = call_gemini_explain(prompt, cmd, [m.full_path for m in file_metas], str(OUTPUT_DIR)) if enable_explain else ""
    t_explain_s = time.perf_counter() - t_explain0 if enable_explain else 0.0

    total_s = time.perf_counter() - t_total_start
    timings = {
        "total_s": total_s,
        "upload_and_hash_s": t_upload_hash_s,
        "ffprobe_s": t_ffprobe_s,
        "gen_iterations_s": [r.gen_s or 0.0 for r in reviews],
        "critic_iterations_s": [r.critic_s or 0.0 for r in reviews],
        "sanitize_iterations_s": [r.sanitize_s or 0.0 for r in reviews],
        "explain_s": t_explain_s,
    }

    notes = [
        "Inputs saved under uploads and rewritten to full paths in the command.",
        "Outputs are enforced to be under the configured default output directory.",
    ]
    if custom_output_abs:
        display_rel = custom_output_rel or Path(custom_output_abs).name
        notes.append(f"Custom output override applied: {display_rel}")

    resp = GenerateResponse(
        prompt=prompt,
        command=cmd,
        explanation=explanation,
        inputs=file_metas,
        default_output_dir=str(OUTPUT_DIR),
        estimated_output_path=est_output,
        safety=safety,
        model="gemini-2.5-flash",
        created_at=datetime.datetime.utcnow().isoformat() + "Z",
        notes=notes,
        reviews=reviews,
        timings=timings,
    )
    return resp


@app.post("/execute")
async def execute_command(command: str = Form(..., description="ffmpeg command to execute")) -> dict:
    cmd = (command or "").strip()
    if not cmd.lower().startswith("ffmpeg"):
        raise HTTPException(status_code=400, detail="Only ffmpeg commands can be executed")
    for tok in FORBIDDEN_TOKENS:
        if tok in cmd:
            raise HTTPException(status_code=400, detail=f"Forbidden token detected in command: {tok}")
    try:
        tokens = shlex.split(cmd)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to parse command: {exc}")
    if not tokens or tokens[0].lower() != "ffmpeg":
        raise HTTPException(status_code=400, detail="Command must start with ffmpeg")

    try:
        result = await run_in_threadpool(
            lambda: subprocess.run(
                tokens,
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=600,
            )
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Command timed out after 600 seconds")

    return {
        "returncode": result.returncode,
        "stdout": (result.stdout or "")[-4000:],
        "stderr": (result.stderr or "")[-4000:],
    }


# Optional: run with `uvicorn main:app --reload`
