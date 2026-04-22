# Realtime STT Hugging Face Test Bench

Standalone extraction of the realtime STT path from the voice-agent project:

- browser microphone capture
- 48 kHz PCM batches over WebSocket
- server-side resampling to 16 kHz
- RealtimeSTT backed by faster-whisper
- SpeechBrain wav2vec2 + CTC support for `aioxlabs/dvoice-darija`
- partial and final transcript updates in the page

## Run

### Changing Models

Change the active model in one place only:

```bash
cp .env.example .env
```

Open `.env` and edit this line:

```env
STT_MODEL_ID=aioxlabs/dvoice-darija
```

Supported values:

```env
# SpeechBrain wav2vec2 + CTC, no CT2 conversion needed
STT_MODEL_ID=aioxlabs/dvoice-darija

# Existing faster-whisper / CTranslate2 model
STT_MODEL_ID=anaszil/whisper-large-v3-turbo-darija
```

Do not edit `Dockerfile` or `docker-compose.yml` just to switch between these
models. Docker Compose reads `STT_MODEL_ID` from `.env`.

The model details live in `stt_config.py` in `MODEL_PRESETS`. To add another
model later, add one preset there, then set `STT_MODEL_ID` in `.env`.

### Docker GPU

The Docker path is the recommended way to run this on an NVIDIA GPU. It pins
Torch and Torchaudio to matching CUDA wheels and preloads the STT model during
FastAPI startup.

The default `.env.example` model is `aioxlabs/dvoice-darija`, loaded through the
SpeechBrain provider. On first use it downloads into `models/speechbrain`.

```bash
cp .env.example .env
mkdir -p models/speechbrain
docker compose up --build
```

Open `http://127.0.0.1:8085` or your cloud forwarded URL for port `8085`.
The app starts on `aioxlabs/dvoice-darija`; the existing faster-whisper model
remains available from the model dropdown if `models/darija` is prepared.

To use the faster-whisper / CTranslate2 model, first convert the Darija LoRA
adapter to a local CTranslate2 model:

```bash
pip install -r requirements-convert.txt
python scripts/convert_darija_lora_to_ct2.py --output-dir models/darija --force
```

Then set this in `.env`:

```env
STT_MODEL_ID=anaszil/whisper-large-v3-turbo-darija
```

For the faster-whisper path, the serving container does not download from
Hugging Face. It expects the converted model at `models/darija`, with files
such as `model.bin`, `config.json`, `tokenizer.json`, `preprocessor_config.json`,
and `vocabulary.json`.

Lightning Studio blocks creating extra virtual environments, so run the
conversion in the default Studio conda environment, or run it in a one-off Docker
container:

```bash
docker run --rm --gpus all \
  -v "$PWD:/work" \
  -w /work \
  pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime \
  bash -lc "pip install -r requirements-convert.txt && python scripts/convert_darija_lora_to_ct2.py --output-dir models/darija --force"
```

If the machine that runs Docker has no outgoing Hugging Face access, run the
conversion on a machine that does have access, then copy the finished
`models/darija` directory into this project.

If your Docker Compose version does not support `gpus: all`, run the image with:

```bash
docker build -t realtime-stt-darija .
docker run --rm --gpus all -p 8085:8085 \
  -v "$PWD/models/darija:/models/darija:ro" \
  -v "$PWD/models/speechbrain:/models/speechbrain" \
  realtime-stt-darija
```

This compose file uses `runtime: nvidia` for compatibility with Compose builds
that reject the newer `gpus` property. If your Docker daemon does not have the
NVIDIA runtime registered, use the `docker run --gpus all` command above.

When the faster-whisper model is selected, startup/session load validates
`/models/darija/model.bin`. If the local CT2 files are missing, the app reports a
clear error instead of calling Hugging Face for the faster-whisper model at
runtime. The SpeechBrain model is downloaded by SpeechBrain from Hugging Face the
first time `aioxlabs/dvoice-darija` is selected.

Check the mounted model before starting Docker:

```bash
ls -lh models/darija/model.bin \
  models/darija/config.json \
  models/darija/tokenizer.json \
  models/darija/preprocessor_config.json \
  models/darija/vocabulary.json
```

### Local Python

```bash
pip install -r requirements.txt
cp .env.example .env
python app.py
```

Open `http://127.0.0.1:8085`, wait for startup to finish loading the configured
model, then press `Start`.

If PyAudio fails to install on macOS, install PortAudio first:

```bash
brew install portaudio
```

## Model Names

This app has explicit backend routing:

- Faster-whisper / RealtimeSTT models use the existing local CTranslate2 path,
  usually `models/darija` mounted as `/models/darija`.
- `aioxlabs/dvoice-darija` uses the SpeechBrain provider and is loaded with
  `EncoderASR.from_hparams`.

The browser sends mono PCM, and the backend resamples it to 16 kHz before either
provider receives it. The SpeechBrain provider writes each detected utterance as
16 kHz single-channel WAV and calls `transcribe_file`.

The original Darija model `anaszil/whisper-large-v3-turbo-darija` is a LoRA/PEFT
adapter, not a directly loadable faster-whisper model. Merge it with
`openai/whisper-large-v3-turbo` and convert the merged model to CTranslate2:

```bash
python scripts/convert_darija_lora_to_ct2.py --output-dir models/darija --force
```

If your base model or adapter is already downloaded locally, pass local paths:

```bash
python scripts/convert_darija_lora_to_ct2.py \
  --base-model /path/to/whisper-large-v3-turbo \
  --adapter /path/to/whisper-large-v3-turbo-darija \
  --tokenizer-source /path/to/whisper-large-v3-turbo \
  --output-dir models/darija \
  --force
```

To start directly with the faster-whisper CT2 model instead of the default
SpeechBrain model:

```bash
STT_MODEL_ID=anaszil/whisper-large-v3-turbo-darija python app.py
```
