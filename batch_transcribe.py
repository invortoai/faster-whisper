
import argparse, os, json, sys, time, pathlib
from tqdm import tqdm
import soundfile as sf
from faster_whisper import WhisperModel

def list_audio_files(root):
    exts = {".wav",".mp3",".m4a",".flac",".ogg",".aac",".wma",".webm",".mp4"}
    for p in sorted(pathlib.Path(root).rglob("*")):
        if p.suffix.lower() in exts and p.is_file():
            yield p

def already_done(out_dir, base):
    return (out_dir / f"{base}.txt").exists() and (out_dir / f"{base}.json").exists()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True, help="Input folder containing audio files")
    ap.add_argument("--out", dest="out_dir", required=True, help="Output folder for transcripts")
    ap.add_argument("--model", default="medium", help="Whisper model name (e.g., base, small, medium, large-v3)")
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"], help="Device selection")
    ap.add_argument("--compute", default=None, help="Compute type: float16, int8_float16, int8, float32")
    ap.add_argument("--lang", default=None, help="Force language code, e.g., hi or en")
    ap.add_argument("--vad", action="store_true", help="Enable VAD filter to skip silence")
    ap.add_argument("--beam", type=int, default=5, help="Beam size (quality vs speed)")
    ap.add_argument("--resume", action="store_true", help="Skip files that already have outputs")
    args = ap.parse_args()

    in_dir = pathlib.Path(args.in_dir)
    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if (args.device=="auto" and os.environ.get("CUDA_VISIBLE_DEVICES", "")!="") else args.device
    if device=="auto": device="cpu"

    model = WhisperModel(args.model, device=device, compute_type=(args.compute or ("float16" if device=="cuda" else "int8")))

    files = list(list_audio_files(in_dir))
    if not files:
        print("No audio files found in", in_dir); sys.exit(1)

    for f in tqdm(files, desc="Transcribing"):
        base = f.stem
        if args.resume and already_done(out_dir, base):
            continue

        # Transcribe
        segments, info = model.transcribe(str(f), language=args.lang, vad_filter=args.vad, beam_size=args.beam)

        # Collect segments
        seg_list = []
        text_all = []
        for s in segments:
            seg_list.append({
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "avg_logprob": s.avg_logprob,
                "no_speech_prob": s.no_speech_prob
            })
            text_all.append(s.text)

        # Write outputs
        (out_dir / f"{base}.txt").write_text("".join(text_all), encoding="utf-8")
        meta = {
            "file": str(f),
            "language": info.language,
            "duration": info.duration,
            "model": args.model,
            "device": device,
            "segments": seg_list
        }
        (out_dir / f"{base}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done. Outputs in", out_dir)

if __name__ == "__main__":
    main()
