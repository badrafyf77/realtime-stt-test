# Realtime STT Hugging Face Test Bench

Standalone extraction of the realtime STT path from the voice-agent project:

- browser microphone capture
- 48 kHz PCM batches over WebSocket
- server-side resampling to 16 kHz
- RealtimeSTT backed by faster-whisper
- partial and final transcript updates in the page

## Run

```bash
cd ~/Downloads/realtime-stt-hf-test
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python app.py
```

Open `http://127.0.0.1:8085`, choose a model, then press `Start`.

The first run downloads the selected model from Hugging Face. If PyAudio fails to
install on macOS, install PortAudio first:

```bash
brew install portaudio
```

## Model Names

The model field accepts faster-whisper model sizes such as:

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

Leave `Realtime` empty to use the same model for realtime partials. Put a smaller
model there when you want faster partial updates while keeping a larger final
model.
