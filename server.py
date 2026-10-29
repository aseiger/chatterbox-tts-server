"""FastAPI TTS server for Chatterbox (resemble-ai).

Endpoints:
  GET  /health   -> server status
  GET  /         -> basic info
  POST /generate -> multipart form:
      text                 (required) text to synthesize
      audio_prompt         (optional) reference wav for voice cloning
      exaggeration         (0.0-1.0, default 0.5)  emotion intensity
      cfg_weight           (default 0.5)
      temperature          (default 0.8)
      repetition_penalty   (default 1.2)
      min_p                (default 0.05)
      top_p                (default 1.0)
    -> audio/wav response (24 kHz)
  POST /tts      -> JSON:
      {"text": "...", "reference_audio_filename": "<voice name>",
       "chunk_size": 80, "stream": true, ...}
    "text" is the only required field. "reference_audio_filename" is a voice
    name (empty or unknown = built-in voice).
    stream=false (default): one standard 24 kHz WAV for the whole text.
    stream=true: response is one 44-byte WAV header (16-bit mono, RIFF and
    data sizes 0xFFFFFFFF because the total is not known up front) followed
    by raw PCM, flushed per piece as each piece (~chunk_size chars, split at
    natural boundaries: sentence > clause > word) finishes generating, so
    playback starts after the first short piece instead of the whole text.
    Contract: exactly one WAV header per HTTP response; the daemon strips
    it per response. Other daemon fields (voice_mode, output_format,
    split_text) are ignored.

Run:
    uvicorn server:app --host 0.0.0.0 --port 8081
"""

import io
import logging
import os
import re
import tempfile
import threading
import time

import numpy as np
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse

from chatterbox import ChatterboxTTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("chatterbox-server")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Voice registry for /tts:
# - Drop a reference clip into VOICES_DIR, e.g. voices/emma.wav (any format
#   soundfile/librosa can read); then reference_audio_filename="emma" or
#   "emma.wav" clones that voice.
# - Or add an explicit alias here: "emma": Path("/srv/voices/emma.wav")
VOICES_DIR = Path(__file__).resolve().parent / "voices"
VOICE_NAMES: dict = {}


def resolve_voice(name) -> Optional[Path]:
    """Map a voice name to a local reference clip; None = built-in voice."""
    name = str(name or "").strip()
    if not name:
        return None
    if name in VOICE_NAMES:
        return VOICE_NAMES[name]
    p = Path(name)
    candidates = [VOICES_DIR / name]
    if p.suffix:
        if p.suffix.lower() != ".wav":
            candidates.append(VOICES_DIR / (p.stem + ".wav"))
    else:
        candidates.append(VOICES_DIR / (name + ".wav"))
    for cand in candidates:
        if cand.is_file():
            return cand
    logger.info("unknown voice %r, using built-in voice", name)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    logger.info("Loading Chatterbox model on %s ...", DEVICE)
    app.state.model = ChatterboxTTS.from_pretrained(device=DEVICE)
    app.state.lock = threading.Lock()
    logger.info("Model ready in %.1fs", time.time() - t0)
    yield
    app.state.model = None


app = FastAPI(title="Chatterbox TTS Server", lifespan=lifespan)


def _model() -> ChatterboxTTS:
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    return model


def _synthesize(model, lock: threading.Lock, text: str, ref_path: Optional[Path],
                **gen_kwargs):
    """Run generation (serialized) and encode a WAV. Returns (bytes, seconds)."""
    t0 = time.time()
    with lock:
        wav = model.generate(
            text,
            audio_prompt_path=str(ref_path) if ref_path else None,
            **gen_kwargs,
        )
    dt = time.time() - t0
    buf = io.BytesIO()
    sf.write(buf, wav.squeeze(0).numpy(), model.sr, format="WAV")
    logger.info("synthesized %d chars in %.1fs", len(text), dt)
    return buf.getvalue(), dt


@app.get("/")
def root():
    return {
        "name": "chatterbox-tts-server",
        "device": DEVICE,
        "sample_rate": 24000,
        "endpoints": ["GET /health", "POST /generate", "POST /tts"],
    }


@app.get("/health")
def health():
    model = getattr(app.state, "model", None)
    return {
        "status": "ok" if model is not None else "loading",
        "device": DEVICE,
        "cuda_free_mb": (
            torch.cuda.mem_get_info()[0] // (1024 * 1024)
            if torch.cuda.is_available()
            else None
        ),
    }


@app.post("/generate")
async def generate(
    text: str = Form(..., description="Text to synthesize"),
    audio_prompt: Optional[UploadFile] = File(
        None, description="Optional reference wav for voice cloning"
    ),
    exaggeration: float = Form(0.5, ge=0.0, le=1.0),
    cfg_weight: float = Form(0.5, ge=0.0, le=2.0),
    temperature: float = Form(0.8, ge=0.1, le=2.0),
    repetition_penalty: float = Form(1.2, ge=0.0, le=2.0),
    min_p: float = Form(0.05, ge=0.0, le=1.0),
    top_p: float = Form(1.0, ge=0.0, le=1.0),
):
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    model = _model()

    tmp_path: Optional[str] = None
    try:
        if audio_prompt is not None and audio_prompt.filename:
            data = await audio_prompt.read()
            if not data:
                raise HTTPException(status_code=400, detail="audio_prompt file is empty")
            suffix = os.path.splitext(audio_prompt.filename)[1] or ".wav"
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp.write(data)
            tmp.close()
            tmp_path = tmp.name

        data, dt = _synthesize(
            model, app.state.lock, text,
            Path(tmp_path) if tmp_path else None,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("generation failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path:
            os.unlink(tmp_path)

    return Response(
        content=data,
        media_type="audio/wav",
        headers={
            "X-Sample-Rate": str(model.sr),
            "X-Generation-Time": f"{dt:.2f}",
        },
    )


def chunk_text(text: str, chunk_size: int):
    """Yield pieces of <= chunk_size chars, splitting at natural boundaries
    (sentence end > clause comma > word). chunk_size <= 0 means one piece."""
    text = text.strip()
    if not text:
        return
    if not chunk_size or chunk_size <= 0 or len(text) <= chunk_size:
        yield text
        return
    for s in re.split(r"(?<=[.!?…])\s+", text):
        s = s.strip()
        if not s:
            continue
        if len(s) <= chunk_size:
            yield s
            continue
        # Sentence too long: try clause boundaries
        for c in re.split(r"(?<=[,;:])\s+", s):
            c = c.strip()
            if not c:
                continue
            if len(c) <= chunk_size:
                yield c
                continue
            # Still too long: word-level fallback
            piece = ""
            for w in c.split():
                while len(w) > chunk_size:  # pathological unbreakable token
                    if piece:
                        yield piece
                        piece = ""
                    piece, w = w[:chunk_size], w[chunk_size:]
                if piece and len(piece) + 1 + len(w) > chunk_size:
                    yield piece
                    piece = w
                else:
                    piece = f"{piece} {w}".strip()
            if piece:
                yield piece


def _wav_header_unknown_length(sample_rate: int) -> bytes:
    """44-byte WAV header (16-bit mono PCM) with RIFF/data sizes 0xFFFFFFFF.
    Decoders that trust the size field read to end of stream; the daemon's
    pass-through strips this header per response."""
    return (
        b"RIFF" + (0xFFFFFFFF).to_bytes(4, "little") + b"WAVE"
        + b"fmt " + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little")                  # PCM
        + (1).to_bytes(2, "little")                  # mono
        + sample_rate.to_bytes(4, "little")
        + (sample_rate * 2).to_bytes(4, "little")    # byte rate
        + (2).to_bytes(2, "little")                  # block align
        + (16).to_bytes(2, "little")                 # bits per sample
        + b"data" + (0xFFFFFFFF).to_bytes(4, "little")
    )


def _wav_to_pcm16(wav) -> bytes:
    """Tensor from model.generate() -> raw little-endian 16-bit PCM bytes."""
    arr = wav.squeeze(0).cpu().numpy()
    if arr.dtype != np.int16:
        arr = np.clip(arr * 32767.0, -32768, 32767).astype("<i2")
    return arr.tobytes()


def _tts_stream(model, lock: threading.Lock, text: str, ref_path: Optional[Path],
                chunk_size: int):
    """Stream the 0xFFFFFFFF WAV header, then raw PCM per piece as each piece
    finishes (contract: one header per HTTP response, no per-piece headers).
    Holds the model lock for the whole stream (single model instance, shared
    conditionals)."""
    with lock:
        if ref_path is not None:
            model.prepare_conditionals(ref_path)  # once, not per piece
        yield _wav_header_unknown_length(model.sr)
        for i, piece in enumerate(chunk_text(text, chunk_size)):
            t0 = time.time()
            wav = model.generate(piece)
            pcm = _wav_to_pcm16(wav)
            logger.info("stream piece %d: %d chars in %.1fs", i, len(piece), time.time() - t0)
            yield pcm


@app.post("/tts")
async def tts(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="expected a JSON body")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="expected a JSON object")

    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail='field "text" (string) is required')

    model = _model()
    ref = resolve_voice(body.get("reference_audio_filename", ""))
    try:
        chunk_size = int(body.get("chunk_size") or 0)
    except (TypeError, ValueError):
        chunk_size = 0
    stream = bool(body.get("stream", False))

    headers = {
        "X-Sample-Rate": str(model.sr),
        "X-Voice": ref.name if ref else "builtin",
    }

    if not stream:
        try:
            data, dt = _synthesize(model, app.state.lock, text, ref)
        except Exception:
            logger.exception("generation failed")
            raise HTTPException(status_code=500, detail="generation failed")
        headers["X-Generation-Time"] = f"{dt:.2f}"
        return Response(content=data, media_type="audio/wav", headers=headers)

    # Streamed: one 0xFFFFFFFF header, then raw PCM flushed per piece.
    return StreamingResponse(
        _tts_stream(model, app.state.lock, text, ref, chunk_size),
        media_type="audio/wav",
        headers=headers,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)
