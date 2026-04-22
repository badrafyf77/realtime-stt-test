FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8085 \
    STT_MODEL=anaszil/whisper-large-v3-turbo-darija-full-ct2 \
    STT_REALTIME_MODEL=anaszil/whisper-large-v3-turbo-darija-full-ct2 \
    STT_USE_MAIN_MODEL_FOR_REALTIME=true \
    STT_LANGUAGE=ar \
    STT_DEVICE=cuda \
    STT_COMPUTE_TYPE=float16 \
    STT_DOWNLOAD_ROOT=/models/faster-whisper \
    HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libsndfile1 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
COPY docker/constraints-cu128.txt docker/constraints-cu128.txt

RUN pip install --upgrade pip setuptools wheel \
    && pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128 torchaudio==2.8.0+cu128 \
    && pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt -c docker/constraints-cu128.txt \
    && python -c "import torch, torchaudio; assert torch.__version__.split('+')[0] == torchaudio.__version__.split('+')[0], (torch.__version__, torchaudio.__version__)"

COPY . .

RUN mkdir -p /models/faster-whisper /models/huggingface /models/torch

EXPOSE 8085

CMD ["python", "app.py"]
