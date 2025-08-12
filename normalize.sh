
#!/usr/bin/env bash
# Normalize any input to 16 kHz mono WAV (PCM 16-bit)
# Usage: ./normalize.sh input.ext output.wav
set -euo pipefail
ffmpeg -y -i "$1" -ac 1 -ar 16000 -c:a pcm_s16le "$2"
