from __future__ import annotations

from typing import Any, Callable, Protocol

from speechbrain_provider import SpeechBrainASRProvider
from stt_config import STT_BACKEND_FASTER_WHISPER, STT_BACKEND_SPEECHBRAIN, infer_stt_backend
from stt_provider import RealtimeSTTProvider


class STTProvider(Protocol):
    def reset_transcription_state(self) -> None: ...

    def transcribe_loop(self) -> None: ...

    def feed_audio(self, chunk: bytes, audio_meta_data: dict[str, Any] | None = None) -> None: ...

    def flush(self) -> None: ...

    def shutdown(self) -> None: ...


def create_stt_provider(
    source_language: str,
    realtime_transcription_callback: Callable[[str], None] | None,
    full_transcription_callback: Callable[[str], None] | None,
    silence_active_callback: Callable[[bool], None] | None,
    on_recording_start_callback: Callable[[], None] | None,
    recorder_config: dict[str, Any] | None,
) -> STTProvider:
    config = recorder_config or {}
    backend = str(config.get("backend") or infer_stt_backend(str(config.get("model") or "")))

    if backend == STT_BACKEND_SPEECHBRAIN:
        return SpeechBrainASRProvider(
            source_language=source_language,
            realtime_transcription_callback=realtime_transcription_callback,
            full_transcription_callback=full_transcription_callback,
            silence_active_callback=silence_active_callback,
            on_recording_start_callback=on_recording_start_callback,
            recorder_config=config,
        )

    if backend == STT_BACKEND_FASTER_WHISPER:
        return RealtimeSTTProvider(
            source_language=source_language,
            realtime_transcription_callback=realtime_transcription_callback,
            full_transcription_callback=full_transcription_callback,
            silence_active_callback=silence_active_callback,
            on_recording_start_callback=on_recording_start_callback,
            recorder_config=config,
        )

    raise ValueError(f"Unsupported STT backend: {backend}")
