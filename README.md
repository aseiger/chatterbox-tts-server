# chatterbox-tts-server

FastAPI TTS server wrapper for [Chatterbox](https://github.com/resemble-ai/chatterbox)
(Resemble AI). Serves text-to-speech over HTTP with voice cloning and chunked
streaming, packaged as a systemd service.

Tested on: Python 3.11, PyTorch 2.6 (cu124), GTX 1070 (8 GB, ~3.2 GB VRAM used,
24 kHz mono output).

## Layout

| File | Purpose |
|---|---|
| `server.py` | The FastAPI app: loads Chatterbox once at startup, exposes `/health`, `/generate`, `/tts` |
| `chatterbox-tts.service` | systemd unit (adjust `User=`/paths to your machine) |
| `voices/` | Drop reference clips here for named voice cloning (see below) |

## Setup

```bash
# 1. Get the model library and this wrapper
git clone https://github.com/resemble-ai/chatterbox.git
cd chatterbox
cp /path/to/chatterbox-tts-server/server.py .
cp -r /path/to/chatterbox-tts-server/voices .

# 2. Environment (uv recommended)
uv venv .venv --python 3.11
uv pip install -e .            # pins torch 2.6.0 etc. (Pascal/sm_61 compatible)
uv pip install fastapi uvicorn # also pulled in via gradio, but be explicit

# 3. First model download (~2.5 GB) + smoke test
.venv/bin/python -c "
from chatterbox import ChatterboxTTS
import torch
m = ChatterboxTTS.from_pretrained(device='cuda')
torch.save(m.generate('Hello.'), '/tmp/smoke.pt')
"

# 4. Run
.venv/bin/uvicorn server:app --host 0.0.0.0 --port 8081

# 5. (Optional) systemd
sudo cp chatterbox-tts.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now chatterbox-tts
```

## Endpoints

### `GET /health`
`{"status": "ok"|"loading", "device": "cuda", "cuda_free_mb": ...}`

### `POST /generate` — multipart form, one complete WAV
| field | default | notes |
|---|---|---|
| `text` | — | required |
| `audio_prompt` | — | optional reference wav upload for voice cloning |
| `exaggeration` | 0.5 | emotion intensity 0–1 |
| `cfg_weight` | 0.5 | |
| `temperature` | 0.8 | |
| `repetition_penalty` | 1.2 | |
| `min_p` | 0.05 | |
| `top_p` | 1.0 | |

### `POST /tts` — JSON, daemon-facing
```json
{"text": "...", "reference_audio_filename": "emma",
 "chunk_size": 80, "stream": true,
 "voice_mode": "...", "output_format": "...", "split_text": true}
```
- `text` is the only required field.
- `reference_audio_filename` is a **voice name**: a file in `voices/`
  (`emma` or `emma.wav`). Empty/unknown names fall back to the built-in
  voice. No restart needed when adding clips.
- `stream: false` (default) → one standard 24 kHz WAV with true RIFF/data sizes.
- `stream: true` → **streaming contract** (below).
- All other fields (`voice_mode`, `output_format`, `split_text`) are accepted
  and ignored.

#### Streaming contract (`stream: true`)
- The response is **exactly one** 44-byte WAV header per HTTP response
  (PCM 16-bit, mono, 24 kHz) with `RIFF` and `data` sizes set to
  `0xFFFFFFFF` — the total length is not known until generation finishes.
  Decoders that trust the size field (browsers, `wave`, libsndfile) read to
  end of stream.
- Then **raw PCM only**, flushed per piece as each piece finishes generating.
  There are no further `RIFF`/`data` markers in the body.
- Text is split into pieces of at most `chunk_size` chars at natural
  boundaries (sentence end > clause comma > word > hard split), so the first
  bytes of audio arrive after one short piece (~2 s) instead of the whole
  text.
- Consumers that want headerless audio should strip the first `data` chunk
  of each response (search for `b"data"`, skip the following 8 bytes).

## Voices

Drop a clean 5–15 s speech clip into `voices/` (wav/mp3/ogg/flac — anything
soundfile reads). It becomes addressable by filename stem in
`reference_audio_filename`. `voices/` is git-ignored except its README;
reference clips stay local.

## Notes

- Single model instance, single lock: generations are serialized. Concurrent
  requests queue (streaming requests hold the lock for their whole stream).
- Voice conditionals are prepared once per request, not per streamed piece.
- `punc_norm` runs per piece: each piece should end with punctuation for
  natural intonation (the splitter prefers sentence boundaries for this).
- Logs: `journalctl -u chatterbox-tts -f` — each streamed piece is logged
  with its generation time (`stream piece N: K chars in X.Xs`).
