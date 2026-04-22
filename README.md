# Realtime STT Hugging Face Test Bench

Standalone extraction of the realtime STT path from the voice-agent project:

- browser microphone capture
- 48 kHz PCM batches over WebSocket
- server-side resampling to 16 kHz
- RealtimeSTT backed by faster-whisper
- partial and final transcript updates in the page

## Run

### Docker GPU

The Docker path is the recommended way to run this on an NVIDIA GPU. It pins
Torch and Torchaudio to matching CUDA wheels and preloads the STT model during
FastAPI startup.

```bash
docker compose up --build
```

Open `http://127.0.0.1:8085` or your cloud forwarded URL for port `8085`.

If your Docker Compose version does not support `gpus: all`, run the image with:

```bash
docker build -t realtime-stt-darija .
docker run --rm --gpus all -p 8085:8085 realtime-stt-darija
```

This compose file uses `runtime: nvidia` for compatibility with Compose builds
that reject the newer `gpus` property. If your Docker daemon does not have the
NVIDIA runtime registered, use the `docker run --gpus all` command above.

The first startup downloads and loads the Darija model, so the server may take a
few minutes before it reports healthy. Model caches are stored in Docker volumes
so later starts are faster.

### Local Python

```bash
cd ~/Downloads/realtime-stt-hf-test
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python app.py
```

Open `http://127.0.0.1:8085`, wait for startup to finish loading the configured
model, then press `Start`.

The first run downloads the configured model from Hugging Face. If PyAudio fails to
install on macOS, install PortAudio first:

```bash
brew install portaudio
```

## Model Names

This app uses RealtimeSTT/faster-whisper, which requires a CTranslate2 model
directory containing `model.bin`.

The original Darija model `anaszil/whisper-large-v3-turbo-darija` is a LoRA/PEFT
adapter, not a directly loadable faster-whisper model. Use the author's full CT2
export instead:

```text
anaszil/whisper-large-v3-turbo-darija-full-ct2
```

If you change the configured model and restart the server, it can be a
faster-whisper model size such as:

```text
tiny.en
base.en
small.en
medium.en
large-v3
```

It also accepts compatible Hugging Face CTranslate2 repos, for example:

```text
Systran/faster-whisper-small.en
Systran/faster-whisper-medium.en
Systran/faster-whisper-large-v3
```

The browser displays the startup model as read-only because this test bench now
loads one shared recorder at application startup.
