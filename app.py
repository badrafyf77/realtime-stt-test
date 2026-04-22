from __future__ import annotations

import asyncio
import json
import logging
import os
import struct
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from audio_processor import AudioInputProcessor
from stt_provider import build_recorder_config

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("realtime_stt_app")

ROOT_DIR = Path(__file__).resolve().parent
STATIC_DIR = ROOT_DIR / "static"
MAX_AUDIO_QUEUE_SIZE = int(os.getenv("MAX_AUDIO_QUEUE_SIZE", "50"))

app = FastAPI(title="Realtime STT Test Bench")
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
    return JSONResponse({"status": "ok", "websocket_path": "/ws"})


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
                await incoming_chunks.put(None)
                break


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

    def send_from_callback(payload: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(outgoing_messages.put_nowait, payload)

    processor: AudioInputProcessor | None = None
    tasks: list[asyncio.Task[Any]] = []

    try:
        recorder_config = build_recorder_config(model=model, realtime_model=realtime_model)
        processor = AudioInputProcessor(
            language=language or os.getenv("STT_LANGUAGE", "en"),
            recorder_config=recorder_config,
            silence_active_callback=lambda active: send_from_callback(
                {"type": "silence", "active": active}
            ),
        )
        processor.realtime_callback = lambda text: send_from_callback(
            {"type": "partial", "content": text}
        )
        processor.final_callback = lambda text: send_from_callback(
            {"type": "final", "content": text}
        )
        processor.recording_start_callback = lambda: send_from_callback(
            {"type": "recording_start"}
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
                "model": recorder_config.get("model"),
                "realtime_model": recorder_config.get("realtime_model_type"),
                "use_main_model_for_realtime": recorder_config.get("use_main_model_for_realtime"),
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
        await incoming_chunks.put(None)
        if processor is not None:
            processor.shutdown()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8085")),
        reload=False,
        log_config=None,
    )
