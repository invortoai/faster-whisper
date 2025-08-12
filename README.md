
# Self-Hosted Whisper (Hindi + English) — Batch & Local API

This kit gives you two options to transcribe audio **without OpenAI API**:

1) **Batch script** (best for files already on your server)  
2) **Local HTTP API** (optional) if you want to POST files to a service

All based on **faster-whisper** (CTranslate2) for speed/accuracy and Hindi/English support.

---

## Quick Start (Batch)

**Requirements (Ubuntu/Debian)**
```bash
sudo apt update && sudo apt install -y ffmpeg python3-pip
pip install -r requirements.txt
```

**Run**
```bash
# Example: auto language, medium model, GPU if available
python batch_transcribe.py --in ./audio --out ./out --model medium --device auto --compute float16

# Force language (optional), e.g. Hindi
python batch_transcribe.py --in ./audio --out ./out --lang hi

# Resume after interruption (skips already-done files)
python batch_transcribe.py --in ./audio --out ./out --resume
```

**Outputs per file**
- `<basename>.txt` — plain text transcript
- `<basename>.json` — segments with timestamps & avg prob

You can safely re-run with `--resume` and it won't re-transcribe already completed files.

---

## Optional: Run a Local API

```bash
pip install -r requirements-api.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Test:
```bash
curl -F "file=@/path/to/sample.wav" http://localhost:8000/transcribe
```

Config via environment vars:
- `WHISPER_MODEL` (default: `medium`)
- `DEVICE` (`auto` | `cpu` | `cuda`), default: `auto`
- `COMPUTE_TYPE` (`float16` | `int8_float16` | `int8` | `float32`), default: auto
- `LANG` (optional fixed language code like `hi` or `en`)

**Docker (API)**
```bash
# Build and run
docker compose up --build -d
# Then POST to http://localhost:8000/transcribe
```

---

## Model recommendations

- Start with **`medium`** if you lack a large GPU. Good for Hindi/English.
- If you have 16GB+ VRAM, try **`large-v3`** for best accuracy.
- CPU-only? Use **`base`** or **`small`** to keep speed reasonable.

**Tips**
- Use `--vad` to avoid long silences.
- Normalize to 16 kHz mono PCM WAV for best results (see `normalize.sh`).

---

## Benchmark (optional)

```bash
python tools/benchmark.py --model medium --minutes 60 --device auto
```

This will simulate 60 minutes of audio by looping a sample and print throughput stats.

---

## Folder layout

```
.
├─ README.md
├─ requirements.txt           # batch
├─ requirements-api.txt       # api
├─ batch_transcribe.py
├─ app.py                     # local API
├─ normalize.sh               # ffmpeg helper
├─ tools/
│  └─ benchmark.py
└─ audio/                     # put your input files here (any audio supported by ffmpeg)
```

---

## Notes

- Everything runs fully offline; no calls to OpenAI.
- Hindi/English code-switching is handled by multilingual models (default).
