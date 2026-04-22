from __future__ import annotations

import copy
import logging
import os
import re
import threading
import time
from difflib import SequenceMatcher
from typing import Any, Callable

import numpy as np

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from RealtimeSTT import AudioToTextRecorder

logger = logging.getLogger(__name__)

INT16_MAX_ABS_VALUE = 32768.0


def _read_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _read_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid %s=%r; using %.3f.", name, value, default)
        return default


def _read_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid %s=%r; using %d.", name, value, default)
        return default


def strip_ending_punctuation(text: str) -> str:
    text = text.rstrip()
    for char in [".", "!", "?", "。"]:
        while text.endswith(char):
            text = text.rstrip(char)
    return text


class TextSimilarity:
    def __init__(self, n_words: int = 5) -> None:
        self.n_words = n_words
        self._punctuation_regex = re.compile(r"[^\w\s]")
        self._whitespace_regex = re.compile(r"\s+")

    def _normalize_text(self, text: str) -> str:
        text = text.lower()
        text = self._punctuation_regex.sub("", text)
        return self._whitespace_regex.sub(" ", text).strip()

    def calculate_similarity(self, text1: str, text2: str) -> float:
        words1 = self._normalize_text(text1).split()[-self.n_words :]
        words2 = self._normalize_text(text2).split()[-self.n_words :]
        return SequenceMatcher(None, " ".join(words1), " ".join(words2), autojunk=False).ratio()


def build_recorder_config(
    model: str | None = None,
    realtime_model: str | None = None,
) -> dict[str, Any]:
    selected_model = (model or os.getenv("STT_MODEL") or "small.en").strip()
    selected_realtime_model = (realtime_model or os.getenv("STT_REALTIME_MODEL") or selected_model).strip()
    use_main_for_realtime = _read_bool_env("STT_USE_MAIN_MODEL_FOR_REALTIME", True)

    if realtime_model:
        use_main_for_realtime = False

    config: dict[str, Any] = {
        "use_microphone": False,
        "spinner": False,
        "model": selected_model,
        "realtime_model_type": selected_realtime_model,
        "use_main_model_for_realtime": use_main_for_realtime,
        "language": os.getenv("STT_LANGUAGE", "en"),
        "silero_sensitivity": _read_float_env("SILERO_SENSITIVITY", 0.25),
        "webrtc_sensitivity": _read_int_env("WEBRTC_SENSITIVITY", 3),
        "post_speech_silence_duration": _read_float_env("POST_SPEECH_SILENCE_DURATION", 0.6),
        "min_length_of_recording": _read_float_env("MIN_LENGTH_OF_RECORDING", 0.4),
        "min_gap_between_recordings": _read_float_env("MIN_GAP_BETWEEN_RECORDINGS", 0.05),
        "enable_realtime_transcription": True,
        "realtime_processing_pause": _read_float_env("REALTIME_PROCESSING_PAUSE", 0.005),
        "silero_use_onnx": True,
        "silero_deactivity_detection": True,
        "early_transcription_on_silence": 0,
        "beam_size": _read_int_env("BEAM_SIZE", 1),
        "beam_size_realtime": _read_int_env("BEAM_SIZE_REALTIME", 1),
        "no_log_file": True,
        "debug_mode": _read_bool_env("STT_DEBUG", False),
        "initial_prompt": "This is a natural conversation. Common words: okay, um, yeah, hello, thanks.",
        "initial_prompt_realtime": "Natural speech with filler words.",
        "faster_whisper_vad_filter": True,
    }

    for env_name, config_name in [
        ("STT_DEVICE", "device"),
        ("STT_COMPUTE_TYPE", "compute_type"),
        ("STT_DOWNLOAD_ROOT", "download_root"),
    ]:
        value = os.getenv(env_name)
        if value:
            config[config_name] = value

    return config


class RealtimeSTTProvider:
    """Standalone version of this project's RealtimeSTT/faster-whisper wrapper."""

    def __init__(
        self,
        source_language: str = "en",
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
        self.recorder_config = copy.deepcopy(recorder_config or build_recorder_config())
        self.recorder_config["language"] = source_language

        self.recorder: AudioToTextRecorder | None = None
        self.realtime_text: str | None = None
        self.stripped_partial_user_text = ""
        self.sentence_end_cache: list[dict[str, Any]] = []
        self.potential_sentences_yielded: list[dict[str, Any]] = []
        self.final_transcription: str | None = None
        self.shutdown_performed = False
        self.silence_active = False
        self.silence_time = 0.0
        self.last_audio_copy: np.ndarray | None = None
        self.text_similarity = TextSimilarity(n_words=5)

        self._create_recorder()

    def _get_recorder_param(self, param_name: str, default: Any = None) -> Any:
        if not self.recorder:
            return default
        return getattr(self.recorder, param_name, default)

    def _set_recorder_param(self, param_name: str, value: Any) -> None:
        if self.recorder:
            setattr(self.recorder, param_name, value)

    def set_silence(self, silence_active: bool) -> None:
        if self.silence_active != silence_active:
            self.silence_active = silence_active
            if self.silence_active_callback:
                self.silence_active_callback(silence_active)

    def get_last_audio_copy(self) -> np.ndarray | None:
        audio_copy = self.get_audio_copy()
        if audio_copy is not None and len(audio_copy) > 0:
            return audio_copy
        return self.last_audio_copy

    def get_audio_copy(self) -> np.ndarray | None:
        if not self.recorder or not hasattr(self.recorder, "frames"):
            return self.last_audio_copy

        try:
            lock = getattr(self.recorder, "frames_lock", threading.Lock())
            with lock:
                frames_data = list(self.recorder.frames)

            if not frames_data:
                return self.last_audio_copy

            full_audio_array = np.frombuffer(b"".join(frames_data), dtype=np.int16)
            if full_audio_array.size == 0:
                return self.last_audio_copy

            self.last_audio_copy = full_audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE
            return self.last_audio_copy
        except Exception:
            logger.exception("Could not copy recorder audio frames.")
            return self.last_audio_copy

    def _create_recorder(self) -> None:
        def start_silence_detection() -> None:
            self.set_silence(True)
            self.silence_time = self._get_recorder_param("speech_end_silence_start", None) or time.time()

        def stop_silence_detection() -> None:
            self.set_silence(False)
            self.silence_time = 0.0

        def start_recording() -> None:
            self.set_silence(False)
            self.silence_time = 0.0
            if self.on_recording_start_callback:
                self.on_recording_start_callback()

        def stop_recording() -> bool:
            self.get_last_audio_copy()
            return False

        def on_partial(text: str | None) -> None:
            if not text:
                return

            self.realtime_text = text
            stripped_text = strip_ending_punctuation(text)
            if stripped_text != self.stripped_partial_user_text:
                self.stripped_partial_user_text = stripped_text
                if self.realtime_transcription_callback:
                    self.realtime_transcription_callback(text)

        active_config = self.recorder_config.copy()
        active_config["on_realtime_transcription_update"] = on_partial
        active_config["on_turn_detection_start"] = start_silence_detection
        active_config["on_turn_detection_stop"] = stop_silence_detection
        active_config["on_recording_start"] = start_recording
        active_config["on_recording_stop"] = stop_recording

        logger.info(
            "Creating AudioToTextRecorder with model=%s realtime_model=%s use_main_for_realtime=%s.",
            active_config.get("model"),
            active_config.get("realtime_model_type"),
            active_config.get("use_main_model_for_realtime"),
        )
        self.recorder = AudioToTextRecorder(**active_config)
        self._set_recorder_param("use_wake_words", False)

    def transcribe_loop(self) -> None:
        def on_final(text: str | None) -> None:
            if not text:
                return

            self.final_transcription = text
            self.sentence_end_cache.clear()
            self.potential_sentences_yielded.clear()
            self.stripped_partial_user_text = ""
            if self.full_transcription_callback:
                self.full_transcription_callback(text)

        if self.recorder and hasattr(self.recorder, "text"):
            self.recorder.text(on_final)

    def feed_audio(self, chunk: bytes, audio_meta_data: dict[str, Any] | None = None) -> None:
        if self.recorder and not self.shutdown_performed:
            self.recorder.feed_audio(chunk)

    def shutdown(self) -> None:
        if self.shutdown_performed:
            return

        self.shutdown_performed = True
        if self.recorder:
            try:
                self.recorder.shutdown()
            except Exception:
                logger.exception("Error during RealtimeSTT recorder shutdown.")
            finally:
                self.recorder = None
