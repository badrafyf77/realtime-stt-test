from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

logger = logging.getLogger(__name__)

STT_BACKEND_FASTER_WHISPER = "faster_whisper"
STT_BACKEND_SPEECHBRAIN = "speechbrain"

DVOICE_DARIJA_ID = "aioxlabs/dvoice-darija"
WHISPER_DARIJA_ID = "anaszil/whisper-large-v3-turbo-darija"
DEFAULT_STT_MODEL_ID = DVOICE_DARIJA_ID
DEFAULT_STT_LANGUAGE = "ar"


@dataclass(frozen=True)
class ModelPreset:
    model_id: str
    backend: str
    model: str
    display_model: str
    realtime_model: str | None = None
    display_realtime_model: str | None = None
    language: str = DEFAULT_STT_LANGUAGE
    speechbrain_savedir: str | None = None


# Add new supported models here. After that, set STT_MODEL_ID in .env.
MODEL_PRESETS: dict[str, ModelPreset] = {
    DVOICE_DARIJA_ID: ModelPreset(
        model_id=DVOICE_DARIJA_ID,
        backend=STT_BACKEND_SPEECHBRAIN,
        model=DVOICE_DARIJA_ID,
        display_model=DVOICE_DARIJA_ID,
        speechbrain_savedir="pretrained_models/asr-wav2vec2-dvoice-dar",
    ),
    WHISPER_DARIJA_ID: ModelPreset(
        model_id=WHISPER_DARIJA_ID,
        backend=STT_BACKEND_FASTER_WHISPER,
        model="/models/darija",
        display_model=WHISPER_DARIJA_ID,
        realtime_model="/models/darija",
        display_realtime_model=WHISPER_DARIJA_ID,
    ),
}

MODEL_ALIASES: dict[str, str] = {
    "/models/darija": WHISPER_DARIJA_ID,
    "./models/darija": WHISPER_DARIJA_ID,
}


@dataclass(frozen=True)
class ModelOption:
    model_id: str
    model: str
    display_model: str
    backend: str
    realtime_model: str | None
    display_realtime_model: str | None
    language: str
    supports_realtime_model: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model": self.model,
            "display_model": self.display_model,
            "backend": self.backend,
            "realtime_model": self.realtime_model,
            "display_realtime_model": self.display_realtime_model,
            "language": self.language,
            "supports_realtime_model": self.supports_realtime_model,
        }


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


def _resolve_model_id(value: str | None) -> str:
    requested = (value or "").strip()
    if not requested:
        requested = (
            os.getenv("STT_MODEL_ID")
            or os.getenv("STT_MODEL")
            or DEFAULT_STT_MODEL_ID
        ).strip()

    return MODEL_ALIASES.get(requested, requested)


def _preset_from_value(value: str | None) -> ModelPreset:
    model_id = _resolve_model_id(value)
    preset = MODEL_PRESETS.get(model_id)
    if preset is not None:
        return preset

    logger.warning(
        "Unknown STT_MODEL_ID=%r; treating it as a direct faster-whisper model path.",
        model_id,
    )
    return ModelPreset(
        model_id=model_id,
        backend=STT_BACKEND_FASTER_WHISPER,
        model=model_id,
        display_model=model_id,
        realtime_model=model_id,
        display_realtime_model=model_id,
    )


def _runtime_model_from_value(value: str) -> str:
    model_id = _resolve_model_id(value)
    preset = MODEL_PRESETS.get(model_id)
    if preset is not None:
        return preset.model
    return value


def infer_stt_backend(model: str | None) -> str:
    return _preset_from_value(model).backend


def build_stt_config(
    model: str | None = None,
    realtime_model: str | None = None,
    language: str | None = None,
) -> dict[str, Any]:
    preset = _preset_from_value(model)
    selected_language = (language or os.getenv("STT_LANGUAGE") or preset.language).strip()

    if preset.backend == STT_BACKEND_SPEECHBRAIN:
        selected_realtime_model = None
        display_realtime_model = None
        use_main_for_realtime = False
    else:
        selected_realtime_model = realtime_model or preset.realtime_model or preset.model
        selected_realtime_model = _runtime_model_from_value(selected_realtime_model)
        display_realtime_model = preset.display_realtime_model or selected_realtime_model
        use_main_for_realtime = _read_bool_env("STT_USE_MAIN_MODEL_FOR_REALTIME", True)
        if realtime_model:
            use_main_for_realtime = False

    speechbrain_savedir = None
    if preset.backend == STT_BACKEND_SPEECHBRAIN:
        speechbrain_savedir = os.getenv("SPEECHBRAIN_SAVEDIR") or preset.speechbrain_savedir

    config: dict[str, Any] = {
        "model_id": preset.model_id,
        "backend": preset.backend,
        "model": preset.model,
        "display_model": preset.display_model,
        "realtime_model_type": selected_realtime_model,
        "display_realtime_model": display_realtime_model,
        "use_main_model_for_realtime": use_main_for_realtime,
        "language": selected_language,
        "sample_rate": 16000,
        "use_microphone": False,
        "spinner": False,
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
        "initial_prompt": "This is a natural Moroccan Darija and Arabic conversation.",
        "initial_prompt_realtime": "Natural Moroccan Darija speech.",
        "faster_whisper_vad_filter": True,
        "speechbrain_savedir": speechbrain_savedir,
        "speechbrain_speech_rms_threshold": _read_float_env(
            "SPEECHBRAIN_SPEECH_RMS_THRESHOLD",
            0.01,
        ),
        "speechbrain_post_speech_silence_duration": _read_float_env(
            "SPEECHBRAIN_POST_SPEECH_SILENCE_DURATION",
            1.2,
        ),
        "speechbrain_min_recording_duration": _read_float_env(
            "SPEECHBRAIN_MIN_RECORDING_DURATION",
            1.0,
        ),
        "speechbrain_pre_speech_padding_duration": _read_float_env(
            "SPEECHBRAIN_PRE_SPEECH_PADDING_DURATION",
            0.25,
        ),
        "speechbrain_max_utterance_duration": _read_float_env(
            "SPEECHBRAIN_MAX_UTTERANCE_DURATION",
            30.0,
        ),
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


def config_cache_key(config: dict[str, Any]) -> tuple[Any, ...]:
    return (
        config.get("model_id"),
        config.get("backend"),
        config.get("model"),
        config.get("realtime_model_type"),
        config.get("language"),
        config.get("device", "auto"),
        config.get("compute_type", "default"),
        config.get("speechbrain_savedir"),
    )


def get_available_model_options(active_config: dict[str, Any]) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    seen: set[str] = set()

    for preset in MODEL_PRESETS.values():
        if preset.model_id in seen:
            continue
        seen.add(preset.model_id)
        options.append(
            ModelOption(
                model_id=preset.model_id,
                model=preset.model,
                display_model=preset.display_model,
                backend=preset.backend,
                realtime_model=preset.realtime_model,
                display_realtime_model=preset.display_realtime_model,
                language=preset.language,
                supports_realtime_model=preset.backend == STT_BACKEND_FASTER_WHISPER,
            ).as_dict()
        )

    active_model_id = str(active_config.get("model_id") or "").strip()
    if active_model_id and active_model_id not in seen:
        options.insert(
            0,
            ModelOption(
                model_id=active_model_id,
                model=str(active_config.get("model") or active_model_id),
                display_model=str(active_config.get("display_model") or active_model_id),
                backend=str(active_config.get("backend") or STT_BACKEND_FASTER_WHISPER),
                realtime_model=active_config.get("realtime_model_type"),
                display_realtime_model=active_config.get("display_realtime_model"),
                language=str(active_config.get("language") or DEFAULT_STT_LANGUAGE),
                supports_realtime_model=active_config.get("backend") == STT_BACKEND_FASTER_WHISPER,
            ).as_dict()
        )

    return options
