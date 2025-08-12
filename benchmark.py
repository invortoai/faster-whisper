
import argparse, time, pathlib, numpy as np, soundfile as sf
from faster_whisper import WhisperModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medium")
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--minutes", type=int, default=10)
    args = ap.parse_args()

    device = args.device
    if device=="auto":
        device = "cuda" if __import__("os").environ.get("CUDA_VISIBLE_DEVICES","") else "cpu"
    compute = "float16" if device=="cuda" else "int8"

    model = WhisperModel(args.model, device=device, compute_type=compute)

    sr = 16000
    # 10 seconds of synthetic noise (not ideal, but avoids shipping large files)
    sample = np.zeros(sr*10, dtype=np.float32)
    sf.write("tmp.wav", sample, sr)

    total_s = args.minutes * 60
    done = 0
    start = time.time()
    while done < total_s:
        segments, info = model.transcribe("tmp.wav", vad_filter=True)
        done += 10
    elapsed = time.time() - start
    rtf = elapsed / total_s
    print(f"Processed {args.minutes} minutes in {elapsed:.2f}s. RTF={rtf:.3f} (lower is better).")

if __name__ == "__main__":
    main()
