
# PropE Self-Hosted Transcriber (Hindi + English)

Transcribe your call recordings **locally** (no OpenAI fees) using **faster-whisper** and optionally **summarize** with OpenAI GPT.
Two ways to run:

1. **Batch mode** — point to a folder of audio files → get transcripts
2. **Local HTTP API** — POST an audio file → get JSON back

---

## Contents (What each file does)

```
repo-root/
├─ README.md                 ← this file
├─ requirements.txt          ← deps for batch mode
├─ requirements-api.txt      ← deps for local API server
├─ batch_transcribe.py       ← batch transcriber (folder → transcripts)
├─ app.py                    ← FastAPI server (/transcribe)
├─ normalize.sh              ← ffmpeg helper (anything → 16 kHz mono WAV)
├─ benchmark.py              ← speed test (Real-Time Factor)
├─ docker-compose.yml        ← run the API with Docker (GPU-ready)
├─ Dockerfile                ← image for API (CUDA runtime base)
├─ audio/                    ← (YOU create) put input audio here for batch mode
└─ out/                      ← (auto-created) transcripts are written here
```

### File-by-file details

* **requirements.txt**
  Minimal Python deps for **batch**: `faster-whisper`, `soundfile`, `numpy`, `tqdm`.

* **requirements-api.txt**
  Python deps for the **API server**: `fastapi`, `uvicorn`, `python-multipart`, plus `faster-whisper`.

* **batch\_transcribe.py**
  Scans `--in` folder for audio, writes transcripts to `--out`.
  Outputs per file:

  * `<name>.txt` — final transcript (plain text)
  * `<name>.json` — segments with timestamps, avg\_logprob, no\_speech\_prob, etc.

* **app.py**
  FastAPI server exposing `POST /transcribe` (multipart upload).
  Returns JSON:

  ```json
  { "language": "hi", "duration": 123.4, "model": "medium", "device": "cuda", "segments": [...], "text": "..." }
  ```

* **normalize.sh**
  One-liner to normalize any media to 16 kHz mono PCM WAV (Whisper’s sweet spot):

  ```bash
  ./normalize.sh input.any output.wav
  ```

* **benchmark.py**
  Measures throughput (**RTF = Real-Time Factor**). Lower is better; RTF < 1 means faster than real-time.

* **docker-compose.yml / Dockerfile**
  Containerized API with CUDA runtime. Use if you prefer Docker and/or GPU machines.

* **audio/** *(YOU create this folder)*
  Your input files for **batch mode** (e.g., `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.aac`, `.webm`, `.mp4`).

* **out/** *(auto-created)*
  Batch outputs land here.

---

## What YOU need to create

* **`audio/`** (for batch mode)
  Create it and drop your recordings inside:

  ```bash
  mkdir -p audio
  cp /path/to/recordings/*.mp3 audio/
  ```

* **`.env`** *(optional but recommended)*
  Put model/device knobs here (used by API; also useful as documentation for batch):

  ```dotenv
  WHISPER_MODEL=medium       # base | small | medium | large-v3
  DEVICE=auto                # auto | cpu | cuda    (API env var)
  COMPUTE_TYPE=              # float16 | int8_float16 | int8 | float32 (API env var)
  LANG=                      # hi | en | leave blank to auto-detect
  ```

> Note: In **batch mode**, you pass these via CLI flags. In **API mode**, you set env vars.

---

## Prerequisites

* **Python 3.8+**
* **ffmpeg**

  ```bash
  sudo apt update && sudo apt install -y ffmpeg
  ```
* **Python deps (batch)**

  ```bash
  pip install -r requirements.txt
  ```
* **Python deps (API)**

  ```bash
  pip install -r requirements-api.txt
  ```

### GPU (optional but recommended)

* NVIDIA driver + CUDA runtime (the Docker image already includes CUDA runtime).
* For non-Docker host usage, verify: `nvidia-smi`.

---

## Usage

### A) Batch mode (folder → transcripts)

1. Put your files in `audio/`:

```bash
mkdir -p audio
cp /path/to/*.wav audio/
```

2. Run:

```bash
# Creates ./out automatically if missing
python batch_transcribe.py \
  --in ./audio \
  --out ./out \
  --model medium \
  --device auto \
  --compute float16 \
  --vad \
  --resume
```

**Common flags**

* `--model` = `base|small|medium|large-v3`
* `--device` = `auto|cpu|cuda`
* `--compute` = `float16|int8_float16|int8|float32`
* `--lang` = `hi|en` (blank → auto-detect)
* `--vad` — skip long silences (faster)
* `--beam` — decoding quality vs speed (default 5)
* `--resume` — skip files already processed

**Outputs**

* `out/<file>.txt`
* `out/<file>.json`

---

### B) Local API

1. Start the server:

```bash
# If using env vars, create .env (API reads WHISPER_* names slightly differently):
# WHISPER_MODEL -> set as WHISPER_MODEL
# DEVICE, COMPUTE_TYPE, LANG apply to API only
uvicorn app:app --host 0.0.0.0 --port 8000
```

2. Test:

```bash
curl -F "file=@/path/to/audio.wav" http://localhost:8000/transcribe
```

**Docker (GPU)**

```bash
docker compose up --build -d
# hits http://localhost:8000/transcribe
```

---

## Choosing model & compute

### Recommended quick picks

* **GPU (≥16 GB VRAM):**
  `--model large-v3 --compute float16`
* **GPU (8–12 GB VRAM):**
  `--model medium --compute int8_float16`
* **CPU only (fastest):**
  `--model small --compute int8`
* **CPU only (best quality, slower):**
  `--model medium --compute float32`

### What is `--compute` / `COMPUTE_TYPE`?

| Option         | What it means (plain English)                 | Use when…                                |
| -------------- | --------------------------------------------- | ---------------------------------------- |
| `float16`      | Half-size numbers, near full accuracy         | Modern GPU with enough VRAM              |
| `int8_float16` | Mostly 8-bit (small) + some 16-bit (accurate) | Smaller GPUs (8–12 GB VRAM)              |
| `int8`         | Compact 8-bit; fastest on CPU                 | CPU or very low VRAM                     |
| `float32`      | Full-size numbers; most accurate but heavy    | CPU with lots of RAM; you prefer quality |
| *(blank)*      | Auto-pick: GPU→float16, CPU→int8              | Easy default                             |

### What is `--beam`?

Beam search width for decoding.

* Higher = better accuracy, slower.
* Lower = faster, less accurate.
  **Defaults to 5** (good balance). Try 8–10 only if you want a tiny quality bump and don’t mind slower runs.

---

## Benchmark (how fast is your box?)

Run:

```bash
python benchmark.py --model medium --minutes 60 --device auto
```

Output example:

```
Processed 60 minutes in 30.0s. RTF=0.008
```

* **RTF < 1** → faster than real-time (good)
* **RTF = 1** → real-time
* **RTF > 1** → slower than real-time (still fine for batch)

Use this to choose the right `--model` and `--compute`.

---

## Normalizing audio (optional but helpful)

If your sources vary a lot, normalize to 16 kHz mono WAV:

```bash
./normalize.sh input.any output.wav
```

---

## Troubleshooting

* **CUDA/GPU not found**

  * Verify `nvidia-smi`.
  * Temporarily force CPU: `--device cpu` (batch) or `DEVICE=cpu` (API).

* **Memory errors on GPU**

  * Drop model size (e.g., `medium` → `small`) or use `--compute int8_float16`.

* **Batch mode finds no files**

  * Ensure `--in` points to a real folder.
  * Create it if missing: `mkdir -p audio` and place files there.

* **Slow on CPU**

  * Use `--model small` and `--compute int8`.
  * Enable `--vad`.

---

## FAQ

**Q: Do I need the `audio/` folder if I’m using the API?**
**A:** No. `audio/` is only for **batch mode**. API takes direct uploads.

**Q: Can I move `benchmark.py` into a `tools/` folder?**
**A:** Yes. It’s standalone. Update your run path accordingly (e.g., `python tools/benchmark.py …`).
GitHub supports folders; if uploading via web UI, create them manually or push via `git`.

**Q: Can I mix Hindi and English in the same call?**
**A:** Yes. Leave language blank to auto-detect code-switching.

**Q: What about summarization?**
**A:** Transcription is local/offline. Summarization remains your choice (OpenAI GPT recommended).
If you want a fully offline setup, replace GPT with a local LLM or a rule-based extractor.

---

## Safety & Licensing

* **Whisper** (and faster-whisper) are MIT-licensed.
* Ensure your data handling complies with your organization’s privacy & security policies.

---

## Quick Start (copy/paste)

**Batch**

```bash
sudo apt update && sudo apt install -y ffmpeg
pip install -r requirements.txt

mkdir -p audio out
python batch_transcribe.py --in ./audio --out ./out --model medium --device auto --compute float16 --vad --resume
```

**API**

```bash
pip install -r requirements-api.txt
uvicorn app:app --host 0.0.0.0 --port 8000
# Test:
curl -F "file=@/path/to/sample.wav" http://localhost:8000/transcribe
```
