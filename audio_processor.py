from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

import numpy as np
from scipy.signal import resample_poly

from stt_provider import RealtimeSTTProvider

logger = logging.getLogger(__name__)


class AudioInputProcessor:
    """Receives browser PCM chunks, resamples them to 16 kHz, and feeds RealtimeSTT."""

    _RESAMPLE_RATIO = 3

    def __init__(
        self,
        language: str = "en",
        silence_active_callback: Callable[[bool], None] | None = None,
        recorder_config: dict[str, Any] | None = None,
    ) -> None:
        self.last_partial_text: str | None = None
        self.realtime_callback: Callable[[str], None] | None = None
        self.final_callback: Callable[[str], None] | None = None
        self.recording_start_callback: Callable[[], None] | None = None
        self.silence_active_callback = silence_active_callback
        self.interrupted = False
        self._transcription_failed = False

        self.transcriber = RealtimeSTTProvider(
            source_language=language,
            realtime_transcription_callback=self._on_partial_transcript,
            full_transcription_callback=self._on_final_transcript,
            on_recording_start_callback=self._on_recording_start,
            silence_active_callback=self._on_silence_active,
            recorder_config=recorder_config,
        )
        self.transcription_task = asyncio.create_task(
            self._run_transcription_loop(),
            name="realtimestt-transcription-loop",
        )
        logger.info("AudioInputProcessor initialized.")

    def _on_partial_transcript(self, text: str) -> None:
        if text != self.last_partial_text:
            self.last_partial_text = text
            if self.realtime_callback:
                self.realtime_callback(text)

    def _on_final_transcript(self, text: str) -> None:
        if self.final_callback:
            self.final_callback(text)

    def _on_recording_start(self) -> None:
        if self.recording_start_callback:
            self.recording_start_callback()

    def _on_silence_active(self, is_active: bool) -> None:
        if self.silence_active_callback:
            self.silence_active_callback(is_active)

    async def _run_transcription_loop(self) -> None:
        while True:
            try:
                await asyncio.to_thread(self.transcriber.transcribe_loop)
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("RealtimeSTT transcription loop failed.")
                self._transcription_failed = True
                break

    def process_audio_chunk(self, raw_bytes: bytes) -> np.ndarray:
        raw_audio = np.frombuffer(raw_bytes, dtype=np.int16)
        if raw_audio.size == 0:
            return np.array([], dtype=np.int16)

        if np.max(np.abs(raw_audio)) == 0:
            expected_len = int(np.ceil(len(raw_audio) / self._RESAMPLE_RATIO))
            return np.zeros(expected_len, dtype=np.int16)

        audio_float32 = raw_audio.astype(np.float32)
        resampled_float = resample_poly(
            audio_float32,
            1,
            self._RESAMPLE_RATIO,
            window=("kaiser", 5.0),
        )
        return np.clip(resampled_float, -32768, 32767).astype(np.int16)

    async def process_chunk_queue(self, audio_queue: asyncio.Queue[dict[str, Any] | None]) -> None:
        while True:
            if self._transcription_failed:
                break

            audio_data = await audio_queue.get()
            if audio_data is None:
                break

            pcm_data = audio_data.pop("pcm")
            processed = self.process_audio_chunk(pcm_data)
            if processed.size == 0 or self.interrupted:
                continue

            self.transcriber.feed_audio(processed.tobytes(), audio_data)

    def shutdown(self) -> None:
        self.transcriber.shutdown()
        if self.transcription_task and not self.transcription_task.done():
            self.transcription_task.cancel()
