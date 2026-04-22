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

DVOICE_DARIJA_MODEL = "aioxlabs/dvoice-darija"
DVOICE_DARIJA_SAVEDIR = "pretrained_models/asr-wav2vec2-dvoice-dar"
CT2_DARIJA_MODEL = "/models/darija"
CT2_DARIJA_DISPLAY_MODEL = "anaszil/whisper-large-v3-turbo-darija"

DEFAULT_STT_MODEL = DVOICE_DARIJA_MODEL
DEFAULT_STT_LANGUAGE = "ar"

SPEECHBRAIN_MODELS: dict[str, dict[str, str]] = {
    DVOICE_DARIJA_MODEL: {
        "savedir": DVOICE_DARIJA_SAVEDIR,
        "language": "ar",
    },
}


@dataclass(frozen=True)
class ModelOption:
    model: str
    display_model: str
    backend: str
    realtime_model: str | None
    display_realtime_model: str | None
    language: str
    supports_realtime_model: bool

    def as_dict(self) -> dict[str, Any]:
        return {
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


def infer_stt_backend(model: str | None) -> str:
    if (model or "").strip() in SPEECHBRAIN_MODELS:
        return STT_BACKEND_SPEECHBRAIN
    return STT_BACKEND_FASTER_WHISPER


def _default_display_model(runtime_model: str, backend: str) -> str:
    configured_model = (os.getenv("STT_MODEL") or DEFAULT_STT_MODEL).strip()
    configured_display = (os.getenv("STT_DISPLAY_MODEL") or "").strip()

    if backend == STT_BACKEND_SPEECHBRAIN:
        return runtime_model
    if runtime_model == CT2_DARIJA_MODEL:
        return CT2_DARIJA_DISPLAY_MODEL
    if configured_display and runtime_model == configured_model:
        return configured_display
    return runtime_model


def _default_display_realtime_model(
    runtime_model: str,
    realtime_model: str | None,
    backend: str,
) -> str | None:
    if backend == STT_BACKEND_SPEECHBRAIN:
        return None

    configured_model = (os.getenv("STT_MODEL") or DEFAULT_STT_MODEL).strip()
    configured_realtime_model = (os.getenv("STT_REALTIME_MODEL") or configured_model).strip()
    configured_display = (os.getenv("STT_DISPLAY_REALTIME_MODEL") or "").strip()

    if realtime_model == CT2_DARIJA_MODEL:
        return CT2_DARIJA_DISPLAY_MODEL
    if configured_display and realtime_model == configured_realtime_model:
        return configured_display
    return realtime_model


def _resolve_runtime_model(requested_model: str) -> str:
    requested_model = requested_model.strip()
    if requested_model in SPEECHBRAIN_MODELS:
        return requested_model
    if requested_model == CT2_DARIJA_DISPLAY_MODEL:
        return CT2_DARIJA_MODEL

    configured_model = (os.getenv("STT_MODEL") or DEFAULT_STT_MODEL).strip()
    configured_display = (os.getenv("STT_DISPLAY_MODEL") or "").strip()
    if configured_display and requested_model == configured_display:
        return configured_model
    return requested_model


def _speechbrain_savedir(model: str) -> str:
    override = os.getenv("SPEECHBRAIN_SAVEDIR")
    if override:
        return override
    return SPEECHBRAIN_MODELS.get(model, {}).get("savedir", f"pretrained_models/{model.replace('/', '--')}")


def build_stt_config(
    model: str | None = None,
    realtime_model: str | None = None,
    language: str | None = None,
) -> dict[str, Any]:
    selected_model = _resolve_runtime_model((model or os.getenv("STT_MODEL") or DEFAULT_STT_MODEL).strip())
    backend = infer_stt_backend(selected_model)
    selected_language = (
        language
        or SPEECHBRAIN_MODELS.get(selected_model, {}).get("language")
        or os.getenv("STT_LANGUAGE")
        or DEFAULT_STT_LANGUAGE
    ).strip()

    if backend == STT_BACKEND_SPEECHBRAIN:
        selected_realtime_model = None
        use_main_for_realtime = False
    else:
        selected_realtime_model = _resolve_runtime_model(
            (realtime_model or os.getenv("STT_REALTIME_MODEL") or selected_model).strip()
        )
        use_main_for_realtime = _read_bool_env("STT_USE_MAIN_MODEL_FOR_REALTIME", True)
        if realtime_model:
            use_main_for_realtime = False

    config: dict[str, Any] = {
        "backend": backend,
        "model": selected_model,
        "display_model": _default_display_model(selected_model, backend),
        "realtime_model_type": selected_realtime_model,
        "display_realtime_model": _default_display_realtime_model(
            selected_model,
            selected_realtime_model,
            backend,
        ),
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
        "speechbrain_savedir": _speechbrain_savedir(selected_model)
        if backend == STT_BACKEND_SPEECHBRAIN
        else None,
        "speechbrain_speech_rms_threshold": _read_float_env(
            "SPEECHBRAIN_SPEECH_RMS_THRESHOLD",
            0.01,
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
        config.get("backend"),
        config.get("model"),
        config.get("realtime_model_type"),
        config.get("language"),
        config.get("device", "auto"),
        config.get("compute_type", "default"),
        config.get("speechbrain_savedir"),
    )


def get_available_model_options(active_config: dict[str, Any]) -> list[dict[str, Any]]:
    options: list[ModelOption] = []
    seen: set[str] = set()

    def add_option(config: dict[str, Any]) -> None:
        model = str(config.get("model") or "").strip()
        if not model or model in seen:
            return

        backend = str(config.get("backend") or infer_stt_backend(model))
        seen.add(model)
        options.append(
            ModelOption(
                model=model,
                display_model=str(config.get("display_model") or model),
                backend=backend,
                realtime_model=config.get("realtime_model_type"),
                display_realtime_model=config.get("display_realtime_model"),
                language=str(config.get("language") or DEFAULT_STT_LANGUAGE),
                supports_realtime_model=backend == STT_BACKEND_FASTER_WHISPER,
            )
        )

    add_option(active_config)

    if active_config.get("model") != CT2_DARIJA_MODEL:
        add_option(build_stt_config(model=CT2_DARIJA_MODEL, realtime_model=CT2_DARIJA_MODEL))

    for speechbrain_model in SPEECHBRAIN_MODELS:
        add_option(build_stt_config(model=speechbrain_model))

    return [option.as_dict() for option in options]
