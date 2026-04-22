FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8085 \
    STT_MODEL=/models/darija \
    STT_REALTIME_MODEL=/models/darija \
    STT_DISPLAY_MODEL=anaszil/whisper-large-v3-turbo-darija \
    STT_DISPLAY_REALTIME_MODEL=anaszil/whisper-large-v3-turbo-darija \
    STT_USE_MAIN_MODEL_FOR_REALTIME=true \
    STT_LANGUAGE=ar \
    STT_DEVICE=cuda \
    STT_COMPUTE_TYPE=float16 \
    STT_DOWNLOAD_ROOT=/models \
    SPEECHBRAIN_SAVEDIR=/models/speechbrain/asr-wav2vec2-dvoice-dar \
    TORCH_HOME=/opt/torch-cache

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

RUN mkdir -p /opt/torch-cache \
    && TORCH_HOME=/opt/torch-cache python -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)"

COPY . .

RUN mkdir -p /models/darija /models/speechbrain

EXPOSE 8085

CMD ["python", "app.py"]
