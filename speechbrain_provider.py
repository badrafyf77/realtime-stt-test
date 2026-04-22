from __future__ import annotations

import logging
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.io import wavfile

logger = logging.getLogger(__name__)

INT16_MAX_ABS_VALUE = 32768.0


class SpeechBrainASRProvider:
    """Buffered SpeechBrain wav2vec2 + CTC ASR provider for 16 kHz mono PCM."""

    def __init__(
        self,
        source_language: str = "ar",
        realtime_transcription_callback: Callable[[str], None] | None = None,
        full_transcription_callback: Callable[[str], None] | None = None,
        silence_active_callback: Callable[[bool], None] | None = None,
        on_recording_start_callback: Callable[[], None] | None = None,
        recorder_config: dict[str, Any] | None = None,
    ) -> None:
        self.source_language = source_language
        self.realtime_transcription_callback = realtime_transcription_callback
        self.full_transcription_callback = full_transcription_callback
        self.silence_active_callback = silence_active_callback
        self.on_recording_start_callback = on_recording_start_callback
        self.recorder_config = recorder_config or {}

        self.model_source = str(self.recorder_config.get("model") or "").strip()
        self.savedir = str(self.recorder_config.get("speechbrain_savedir") or "").strip()
        self.sample_rate = int(self.recorder_config.get("sample_rate") or 16000)
        self.speech_rms_threshold = float(
            self.recorder_config.get("speechbrain_speech_rms_threshold", 0.01)
        )
        self.post_speech_silence_duration = float(
            self.recorder_config.get("post_speech_silence_duration", 0.6)
        )
        self.min_recording_duration = float(
            self.recorder_config.get("min_length_of_recording", 0.4)
        )
        self.pre_speech_padding_duration = float(
            self.recorder_config.get("speechbrain_pre_speech_padding_duration", 0.25)
        )
        self.max_utterance_duration = float(
            self.recorder_config.get("speechbrain_max_utterance_duration", 30.0)
        )

        self.asr_model: Any | None = None
        self.shutdown_performed = False
        self.silence_active = False
        self.final_transcription: str | None = None
        self.last_audio_copy: np.ndarray | None = None

        self._condition = threading.Condition()
        self._utterance_queue: deque[np.ndarray] = deque()
        self._pre_speech_chunks: deque[np.ndarray] = deque()
        self._pre_speech_sample_count = 0
        self._recorded_chunks: list[np.ndarray] = []
        self._recording_sample_count = 0
        self._is_recording = False
        self._silence_sample_count = 0

        self._load_model()

    def _load_model(self) -> None:
        if not self.model_source:
            raise RuntimeError("SpeechBrain ASR model source is empty.")

        try:
            from speechbrain.inference.ASR import EncoderASR
        except ImportError:
            from speechbrain.pretrained import EncoderASR

        run_opts: dict[str, Any] = {}
        configured_device = str(self.recorder_config.get("device", "")).strip()
        if configured_device and configured_device.lower() != "auto":
            run_opts["device"] = configured_device

        self._log_torch_cuda_status()
        start_time = time.perf_counter()
        logger.info(
            "Loading SpeechBrain provider model=%s savedir=%s run_opts=%s.",
            self.model_source,
            self.savedir,
            run_opts or {},
        )
        load_kwargs: dict[str, Any] = {
            "source": self.model_source,
            "savedir": self.savedir,
        }
        if run_opts:
            load_kwargs["run_opts"] = run_opts
        self.asr_model = EncoderASR.from_hparams(**load_kwargs)
        logger.info(
            "SpeechBrain provider loaded successfully model=%s savedir=%s load_time=%.3fs.",
            self.model_source,
            self.savedir,
            time.perf_counter() - start_time,
        )

    def _log_torch_cuda_status(self) -> None:
        try:
            import torch
        except Exception:
            logger.info("CUDA available to torch: false; torch import failed.")
            return

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            logger.info(
                "CUDA available to torch: true device=%s torch=%s cuda=%s.",
                torch.cuda.get_device_name(0),
                torch.__version__,
                torch.version.cuda,
            )
        else:
            logger.info("CUDA available to torch: false torch=%s.", torch.__version__)

    def reset_transcription_state(self) -> None:
        with self._condition:
            self.final_transcription = None
            self._utterance_queue.clear()
            self._recorded_chunks.clear()
            self._recording_sample_count = 0
            self._is_recording = False
            self._silence_sample_count = 0
            self.set_silence(False)

    def set_silence(self, silence_active: bool) -> None:
        if self.silence_active != silence_active:
            self.silence_active = silence_active
            if self.silence_active_callback:
                self.silence_active_callback(silence_active)

    def get_last_audio_copy(self) -> np.ndarray | None:
        if self._recorded_chunks:
            return np.concatenate(self._recorded_chunks)
        return self.last_audio_copy

    def _remember_pre_speech_audio(self, samples: np.ndarray) -> None:
        max_samples = int(self.pre_speech_padding_duration * self.sample_rate)
        if max_samples <= 0:
            return

        self._pre_speech_chunks.append(samples)
        self._pre_speech_sample_count += len(samples)
        while self._pre_speech_sample_count > max_samples and self._pre_speech_chunks:
            removed = self._pre_speech_chunks.popleft()
            self._pre_speech_sample_count -= len(removed)

    def _start_recording(self) -> None:
        if self._is_recording:
            return

        self._is_recording = True
        self._silence_sample_count = 0
        self._recorded_chunks = list(self._pre_speech_chunks)
        self._recording_sample_count = sum(len(chunk) for chunk in self._recorded_chunks)
        self.set_silence(False)
        if self.on_recording_start_callback:
            self.on_recording_start_callback()

    def _queue_recording(self) -> None:
        if not self._recorded_chunks:
            self._reset_recording()
            return

        duration = self._recording_sample_count / self.sample_rate
        audio = np.concatenate(self._recorded_chunks)
        self.last_audio_copy = audio
        self._reset_recording()

        if duration < self.min_recording_duration:
            logger.debug(
                "Discarding short SpeechBrain utterance duration=%.3fs min=%.3fs.",
                duration,
                self.min_recording_duration,
            )
            return

        with self._condition:
            self._utterance_queue.append(audio)
            self._condition.notify()

    def _reset_recording(self) -> None:
        self._is_recording = False
        self._recorded_chunks = []
        self._recording_sample_count = 0
        self._silence_sample_count = 0
        self.set_silence(False)

    def feed_audio(self, chunk: bytes, audio_meta_data: dict[str, Any] | None = None) -> None:
        if self.shutdown_performed:
            return

        samples_int16 = np.frombuffer(chunk, dtype=np.int16)
        if samples_int16.size == 0:
            return

        samples = samples_int16.astype(np.float32) / INT16_MAX_ABS_VALUE
        rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
        has_speech = rms >= self.speech_rms_threshold

        if not self._is_recording:
            self._remember_pre_speech_audio(samples)
            if not has_speech:
                return
            self._start_recording()

        self._recorded_chunks.append(samples)
        self._recording_sample_count += len(samples)

        if has_speech:
            self._silence_sample_count = 0
            self.set_silence(False)
        else:
            self._silence_sample_count += len(samples)
            self.set_silence(True)

        silence_duration = self._silence_sample_count / self.sample_rate
        recording_duration = self._recording_sample_count / self.sample_rate
        if (
            silence_duration >= self.post_speech_silence_duration
            or recording_duration >= self.max_utterance_duration
        ):
            self._queue_recording()

    def flush(self) -> None:
        if self._is_recording:
            self._queue_recording()

    def transcribe_loop(self) -> None:
        with self._condition:
            while not self._utterance_queue and not self.shutdown_performed:
                self._condition.wait(timeout=0.5)
            if self.shutdown_performed:
                return
            audio = self._utterance_queue.popleft()

        duration = len(audio) / self.sample_rate
        start_time = time.perf_counter()
        text = self._transcribe_audio(audio)
        inference_time = time.perf_counter() - start_time

        if not text:
            logger.info(
                "SpeechBrain transcription produced empty output duration=%.3fs inference_time=%.3fs.",
                duration,
                inference_time,
            )
            return

        self.final_transcription = text
        logger.info(
            "SpeechBrain transcription output=%r duration=%.3fs inference_time=%.3fs.",
            text,
            duration,
            inference_time,
        )
        if self.full_transcription_callback:
            self.full_transcription_callback(text)

    def _transcribe_audio(self, audio: np.ndarray) -> str:
        if self.asr_model is None:
            raise RuntimeError("SpeechBrain ASR model is not loaded.")

        audio_int16 = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio_int16 * 32767.0).astype(np.int16)
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            wavfile.write(temp_path, self.sample_rate, audio_int16)
            result = self.asr_model.transcribe_file(temp_path)
        finally:
            if temp_path:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except OSError:
                    logger.warning("Could not remove temporary SpeechBrain audio file: %s.", temp_path)

        if isinstance(result, str):
            return result.strip()
        return str(result).strip()

    def shutdown(self) -> None:
        if self.shutdown_performed:
            return

        self.shutdown_performed = True
        with self._condition:
            self._utterance_queue.clear()
            self._condition.notify_all()
