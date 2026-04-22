from __future__ import annotations

import asyncio
import json
import logging
import os
import struct
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from audio_processor import AudioInputProcessor
from stt_config import (
    STT_BACKEND_FASTER_WHISPER,
    build_stt_config,
    get_available_model_options,
)
from stt_manager import STTProcessorManager

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("realtime_stt_app")

ROOT_DIR = Path(__file__).resolve().parent
STATIC_DIR = ROOT_DIR / "static"
MAX_AUDIO_QUEUE_SIZE = int(os.getenv("MAX_AUDIO_QUEUE_SIZE", "50"))
REQUIRED_CT2_FILES = (
    "model.bin",
    "config.json",
    "tokenizer.json",
    "preprocessor_config.json",
    "vocabulary.json",
)


def _display_model_name(recorder_config: dict[str, Any]) -> str | None:
    return recorder_config.get("display_model") or recorder_config.get("model")


def _display_realtime_model_name(recorder_config: dict[str, Any]) -> str | None:
    return recorder_config.get("display_realtime_model") or recorder_config.get("realtime_model_type")


def _log_cuda_status() -> bool:
    try:
        import torch
    except Exception as exc:
        logger.warning("CUDA available: false; torch could not be imported: %s", exc)
        return False

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        logger.info(
            "CUDA available: true device=%s torch=%s cuda=%s.",
            torch.cuda.get_device_name(0),
            torch.__version__,
            torch.version.cuda,
        )
    else:
        logger.info("CUDA available: false torch=%s.", torch.__version__)
    return cuda_available


def _assert_cuda_if_requested(recorder_config: dict[str, Any]) -> None:
    if str(recorder_config.get("device", "")).lower() != "cuda":
        _log_cuda_status()
        return

    if not _log_cuda_status():
        raise RuntimeError(
            "STT_DEVICE=cuda was requested, but torch cannot see a CUDA GPU. "
            "Start the container with the NVIDIA runtime or docker run --gpus all."
        )


def _validate_local_ct2_model(recorder_config: dict[str, Any]) -> None:
    if recorder_config.get("backend") != STT_BACKEND_FASTER_WHISPER:
        return

    paths = {
        str(recorder_config.get("model", "")).strip(),
        str(recorder_config.get("realtime_model_type", "")).strip(),
    }
    paths.discard("")

    for raw_path in paths:
        model_path = Path(raw_path).expanduser()
        missing_files = [
            filename for filename in REQUIRED_CT2_FILES if not (model_path / filename).is_file()
        ]
        if missing_files:
            raise RuntimeError(
                "Local CTranslate2 model is not ready at "
                f"{model_path}. Missing: {', '.join(missing_files)}. "
                "Convert the Darija LoRA model first and mount it at /models/darija."
            )


def _validate_stt_config(recorder_config: dict[str, Any]) -> None:
    _assert_cuda_if_requested(recorder_config)
    _validate_local_ct2_model(recorder_config)


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    recorder_config = build_stt_config()
    language = str(recorder_config.get("language") or "ar")

    logger.info(
        "Preloading STT processor at startup backend=%s model=%s realtime_model=%s language=%s device=%s compute_type=%s.",
        recorder_config.get("backend"),
        recorder_config.get("model"),
        recorder_config.get("realtime_model_type"),
        language,
        recorder_config.get("device", "auto"),
        recorder_config.get("compute_type", "default"),
    )

    manager = STTProcessorManager(validate_config=_validate_stt_config)
    await manager.preload(recorder_config)
    fastapi_app.state.processor_manager = manager
    fastapi_app.state.session_lock = asyncio.Lock()
    fastapi_app.state.active_session = None

    try:
        yield
    finally:
        logger.info("Shutting down STT processor.")
        manager.shutdown()


app = FastAPI(title="Realtime STT Test Bench", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> JSONResponse:
    manager: STTProcessorManager | None = getattr(app.state, "processor_manager", None)
    recorder_config = manager.config if manager and manager.config else build_stt_config()
    return JSONResponse(
        {
            "status": "ok",
            "websocket_path": "/ws",
            "model_loaded": bool(manager and manager.processor),
            "model_id": recorder_config.get("model_id"),
            "backend": recorder_config.get("backend"),
            "model": _display_model_name(recorder_config),
            "model_path": recorder_config.get("model"),
            "realtime_model": _display_realtime_model_name(recorder_config),
            "realtime_model_path": recorder_config.get("realtime_model_type"),
            "language": recorder_config.get("language", "ar"),
            "device": recorder_config.get("device", "auto"),
            "compute_type": recorder_config.get("compute_type", "default"),
        }
    )


@app.get("/config")
async def config() -> JSONResponse:
    manager: STTProcessorManager | None = getattr(app.state, "processor_manager", None)
    recorder_config = manager.config if manager and manager.config else build_stt_config()
    return JSONResponse(
        {
            "backend": recorder_config.get("backend"),
            "model_id": recorder_config.get("model_id"),
            "model": recorder_config.get("model"),
            "display_model": _display_model_name(recorder_config),
            "realtime_model": recorder_config.get("realtime_model_type"),
            "display_realtime_model": _display_realtime_model_name(recorder_config),
            "use_main_model_for_realtime": recorder_config.get("use_main_model_for_realtime"),
            "language": recorder_config.get("language", "ar"),
            "device": recorder_config.get("device", "auto"),
            "compute_type": recorder_config.get("compute_type", "default"),
            "models": get_available_model_options(recorder_config),
        }
    )


def _format_timestamp_ns(timestamp_ns: int) -> str:
    seconds = timestamp_ns // 1_000_000_000
    remainder_ns = timestamp_ns % 1_000_000_000
    local_time = time.localtime(seconds)
    return time.strftime("%H:%M:%S", local_time) + f".{remainder_ns // 1_000_000:03d}"


async def _receive_audio(ws: WebSocket, incoming_chunks: asyncio.Queue[dict[str, Any] | None]) -> None:
    while True:
        message = await ws.receive()

        if raw := message.get("bytes"):
            if len(raw) < 8:
                logger.warning("Received an audio packet shorter than the 8-byte header.")
                continue

            timestamp_ms, flags = struct.unpack("!II", raw[:8])
            client_sent_ns = timestamp_ms * 1_000_000
            server_received_ns = time.time_ns()
            metadata = {
                "client_sent_ms": timestamp_ms,
                "client_sent": client_sent_ns,
                "client_sent_formatted": _format_timestamp_ns(client_sent_ns),
                "server_received": server_received_ns,
                "server_received_formatted": _format_timestamp_ns(server_received_ns),
                "isTTSPlaying": bool(flags & 1),
                "pcm": raw[8:],
            }

            if incoming_chunks.qsize() >= MAX_AUDIO_QUEUE_SIZE:
                logger.warning("Audio queue is full; dropping a chunk.")
                continue

            await incoming_chunks.put(metadata)
            continue

        if text := message.get("text"):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                logger.warning("Ignoring invalid JSON text message from client.")
                continue

            if payload.get("type") == "stop":
                _stop_audio_queue(incoming_chunks)
                break


def _stop_audio_queue(incoming_chunks: asyncio.Queue[dict[str, Any] | None]) -> None:
    try:
        incoming_chunks.put_nowait(None)
    except asyncio.QueueFull:
        try:
            incoming_chunks.get_nowait()
        except asyncio.QueueEmpty:
            pass
        incoming_chunks.put_nowait(None)


async def _send_messages(ws: WebSocket, outgoing_messages: asyncio.Queue[dict[str, Any]]) -> None:
    while True:
        payload = await outgoing_messages.get()
        await ws.send_json(payload)


@app.websocket("/ws")
async def websocket_endpoint(
    ws: WebSocket,
    model: str | None = None,
    realtime_model: str | None = None,
    language: str | None = None,
) -> None:
    await ws.accept()

    loop = asyncio.get_running_loop()
    incoming_chunks: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(MAX_AUDIO_QUEUE_SIZE)
    outgoing_messages: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    session_id = str(id(ws))

    def send_from_callback(payload: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(outgoing_messages.put_nowait, payload)

    tasks: list[asyncio.Task[Any]] = []

    try:
        manager: STTProcessorManager = app.state.processor_manager
        session_lock: asyncio.Lock = app.state.session_lock

        async with session_lock:
            if app.state.active_session is not None:
                await ws.send_json(
                    {
                        "type": "status",
                        "status": "error",
                        "message": "Another browser session is already using the preloaded recorder.",
                    }
                )
                await ws.close(code=1013)
                return
            app.state.active_session = session_id

        requested_config = build_stt_config(
            model=model,
            realtime_model=realtime_model,
            language=language,
        )
        await ws.send_json(
            {
                "type": "status",
                "status": "loading",
                "message": f"Loading {_display_model_name(requested_config)}",
                "model": _display_model_name(requested_config),
                "model_id": requested_config.get("model_id"),
                "model_path": requested_config.get("model"),
                "backend": requested_config.get("backend"),
            }
        )
        processor: AudioInputProcessor = await manager.get_processor(requested_config)
        recorder_config = manager.config or requested_config

        processor.prepare_session(
            realtime_callback=lambda text: send_from_callback({"type": "partial", "content": text}),
            final_callback=lambda text: send_from_callback({"type": "final", "content": text}),
            recording_start_callback=lambda: send_from_callback({"type": "recording_start"}),
            silence_active_callback=lambda active: send_from_callback(
                {"type": "silence", "active": active}
            ),
        )

        tasks = [
            asyncio.create_task(_receive_audio(ws, incoming_chunks), name="receive-audio"),
            asyncio.create_task(processor.process_chunk_queue(incoming_chunks), name="process-audio"),
            asyncio.create_task(_send_messages(ws, outgoing_messages), name="send-messages"),
        ]

        await ws.send_json(
            {
                "type": "status",
                "status": "ready",
                "message": "Ready",
                "model": _display_model_name(recorder_config),
                "model_id": recorder_config.get("model_id"),
                "model_path": recorder_config.get("model"),
                "backend": recorder_config.get("backend"),
                "realtime_model": _display_realtime_model_name(recorder_config),
                "realtime_model_path": recorder_config.get("realtime_model_type"),
                "use_main_model_for_realtime": recorder_config.get("use_main_model_for_realtime"),
                "preloaded": True,
            }
        )

        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            exception = task.exception()
            if exception and not isinstance(exception, asyncio.CancelledError):
                raise exception
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    except WebSocketDisconnect:
        logger.info("Browser disconnected.")
    except Exception as exc:
        logger.exception("WebSocket session failed: %s", exc)
        try:
            await ws.send_json({"type": "status", "status": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        _stop_audio_queue(incoming_chunks)
        manager = getattr(app.state, "processor_manager", None)
        if manager is not None:
            manager.clear_session_callbacks()
        session_lock = getattr(app.state, "session_lock", None)
        if session_lock is not None:
            async with session_lock:
                if getattr(app.state, "active_session", None) == session_id:
                    app.state.active_session = None
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8085")),
        reload=False,
        log_config=None,
    )
