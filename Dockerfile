
FROM nvidia/cuda:12.3.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends     python3 python3-pip ffmpeg &&     rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-api.txt /app/requirements-api.txt
RUN pip3 install --no-cache-dir -r requirements-api.txt

COPY app.py /app/app.py
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
