from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

from audio_processor import AudioInputProcessor
from stt_config import config_cache_key

logger = logging.getLogger(__name__)

ConfigValidator = Callable[[dict[str, Any]], None]


class STTProcessorManager:
    """Owns the single loaded STT processor and switches it when the model changes."""

    def __init__(self, validate_config: ConfigValidator) -> None:
        self._validate_config = validate_config
        self._lock = asyncio.Lock()
        self.processor: AudioInputProcessor | None = None
        self.config: dict[str, Any] | None = None
        self._active_key: tuple[Any, ...] | None = None

    async def preload(self, config: dict[str, Any]) -> AudioInputProcessor:
        return await self.get_processor(config)

    async def get_processor(self, config: dict[str, Any]) -> AudioInputProcessor:
        requested_key = config_cache_key(config)
        async with self._lock:
            if self.processor is not None and self._active_key == requested_key:
                return self.processor

            old_processor = self.processor
            self.processor = None
            self.config = None
            self._active_key = None
            if old_processor is not None:
                logger.info("Shutting down existing STT processor before model switch.")
                old_processor.shutdown()

            self._validate_config(config)
            logger.info(
                "Loading STT processor backend=%s model=%s realtime_model=%s language=%s device=%s.",
                config.get("backend"),
                config.get("model"),
                config.get("realtime_model_type"),
                config.get("language"),
                config.get("device", "auto"),
            )
            start_time = time.perf_counter()
            processor = AudioInputProcessor(
                language=str(config.get("language") or "ar"),
                recorder_config=config,
            )
            self.processor = processor
            self.config = config
            self._active_key = requested_key
            logger.info(
                "STT processor loaded successfully backend=%s model=%s load_time=%.3fs.",
                config.get("backend"),
                config.get("model"),
                time.perf_counter() - start_time,
            )
            return processor

    def clear_session_callbacks(self) -> None:
        if self.processor is not None:
            self.processor.clear_session_callbacks()

    def shutdown(self) -> None:
        if self.processor is not None:
            self.processor.shutdown()
        self.processor = None
        self.config = None
        self._active_key = None
