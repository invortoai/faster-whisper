
import os, tempfile, json
from fastapi import FastAPI, UploadFile, File, Form
from faster_whisper import WhisperModel

model_name = os.getenv("WHISPER_MODEL", "medium")
device     = os.getenv("DEVICE", "auto")
compute    = os.getenv("COMPUTE_TYPE", None)
lang_fixed = os.getenv("LANG", None)

if device == "auto":
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES","") else "cpu"
if compute is None:
    compute = "float16" if device == "cuda" else "int8"

model = WhisperModel(model_name, device=device, compute_type=compute)
app = FastAPI()

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), lang: str | None = Form(None), vad: bool = Form(True), beam: int = Form(5)):
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    segments, info = model.transcribe(tmp_path, language=(lang or lang_fixed), vad_filter=vad, beam_size=beam)
    os.remove(tmp_path)

    segs = [{
        "start": s.start,
        "end": s.end,
        "text": s.text,
        "avg_logprob": s.avg_logprob,
        "no_speech_prob": s.no_speech_prob
    } for s in segments]

    return {
        "language": info.language,
        "duration": info.duration,
        "model": model_name,
        "device": device,
        "segments": segs,
        "text": "".join([s["text"] for s in segs])
    }
