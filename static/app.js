const AUDIO_BATCH_SAMPLES = 2048;
const AUDIO_HEADER_BYTES = 8;
const AUDIO_MESSAGE_BYTES = AUDIO_HEADER_BYTES + AUDIO_BATCH_SAMPLES * 2;

const modelInput = document.querySelector("#model");
const realtimeModelInput = document.querySelector("#realtime-model");
const languageInput = document.querySelector("#language");
const startButton = document.querySelector("#start-button");
const stopButton = document.querySelector("#stop-button");
const clearButton = document.querySelector("#clear-button");
const statusText = document.querySelector("#status-text");
const statusDot = document.querySelector("#status-dot");
const transcript = document.querySelector("#transcript");

let socket = null;
let audioContext = null;
let workletNode = null;
let mediaStream = null;
let batchBuffer = null;
let batchView = null;
let batchSamples = null;
let batchOffset = 0;
let finalLines = [];
let partialLine = "";
let captureStarted = false;

function setStatus(text, state = "") {
  statusText.textContent = text;
  statusDot.className = `status-dot ${state}`.trim();
}

function renderTranscript() {
  transcript.innerHTML = "";

  if (finalLines.length > 0) {
    transcript.append(document.createTextNode(finalLines.join("\n")));
  }

  if (partialLine) {
    if (finalLines.length > 0) {
      transcript.append(document.createTextNode("\n"));
    }
    const partial = document.createElement("span");
    partial.className = "partial";
    partial.textContent = partialLine;
    transcript.append(partial);
  }
}

function buildWebSocketUrl() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return new URL(`${protocol}//${window.location.host}/ws`).toString();
}

function initializeBatch() {
  if (batchBuffer) {
    return;
  }

  batchBuffer = new ArrayBuffer(AUDIO_MESSAGE_BYTES);
  batchView = new DataView(batchBuffer);
  batchSamples = new Int16Array(batchBuffer, AUDIO_HEADER_BYTES);
  batchOffset = 0;
}

function flushBatch() {
  if (!batchBuffer || !batchView || !batchSamples) {
    return;
  }

  batchView.setUint32(0, Date.now() & 0xffffffff, false);
  batchView.setUint32(4, 0, false);

  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(batchBuffer);
  }

  batchBuffer = null;
  batchView = null;
  batchSamples = null;
  batchOffset = 0;
}

function flushRemainder() {
  if (!batchSamples || batchOffset === 0) {
    return;
  }

  for (let index = batchOffset; index < AUDIO_BATCH_SAMPLES; index += 1) {
    batchSamples[index] = 0;
  }
  flushBatch();
}

async function startCapture() {
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      sampleRate: { ideal: 48000 },
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
    },
  });

  audioContext = new AudioContext({ sampleRate: 48000 });
  await audioContext.audioWorklet.addModule("/static/pcmWorkletProcessor.js");
  await audioContext.resume();

  workletNode = new AudioWorkletNode(audioContext, "pcm-worklet-processor");
  workletNode.port.onmessage = ({ data }) => {
    const incoming = new Int16Array(data);
    let readOffset = 0;

    while (readOffset < incoming.length) {
      initializeBatch();

      const copyLength = Math.min(
        incoming.length - readOffset,
        AUDIO_BATCH_SAMPLES - batchOffset,
      );
      batchSamples.set(incoming.subarray(readOffset, readOffset + copyLength), batchOffset);

      batchOffset += copyLength;
      readOffset += copyLength;

      if (batchOffset === AUDIO_BATCH_SAMPLES) {
        flushBatch();
      }
    }
  };

  audioContext.createMediaStreamSource(mediaStream).connect(workletNode);
}

function stopCapture() {
  flushRemainder();
  workletNode?.disconnect();
  workletNode = null;

  mediaStream?.getTracks().forEach((track) => track.stop());
  mediaStream = null;

  if (audioContext) {
    void audioContext.close();
  }
  audioContext = null;
}

function stopSession() {
  stopCapture();
  captureStarted = false;
  if (socket?.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "stop" }));
  }
  socket?.close();
  socket = null;
  startButton.disabled = false;
  stopButton.disabled = true;
  setStatus("Idle");
}

function handleMessage(payload) {
  if (payload.type === "status") {
    if (payload.status === "ready") {
      setStatus(`Ready: ${payload.model}`, "ready");
      if (!captureStarted) {
        captureStarted = true;
        startCapture()
          .then(() => setStatus("Listening", "ready"))
          .catch((error) => {
            stopSession();
            setStatus(error instanceof Error ? error.message : "Microphone error", "error");
          });
      }
    } else if (payload.status === "error") {
      setStatus(payload.message || "Error", "error");
    } else {
      setStatus(payload.message || payload.status || "Connected");
    }
    return;
  }

  if (payload.type === "recording_start") {
    setStatus("Listening", "ready");
    return;
  }

  if (payload.type === "partial") {
    partialLine = payload.content || "";
    renderTranscript();
    return;
  }

  if (payload.type === "final") {
    const content = (payload.content || "").trim();
    if (content && finalLines[finalLines.length - 1] !== content) {
      finalLines.push(content);
    }
    partialLine = "";
    renderTranscript();
  }
}

async function startSession() {
  startButton.disabled = true;
  stopButton.disabled = false;
  setStatus("Connecting");

  try {
    socket = new WebSocket(buildWebSocketUrl());
    socket.binaryType = "arraybuffer";

    socket.onmessage = ({ data }) => {
      if (typeof data === "string") {
        handleMessage(JSON.parse(data));
      }
    };

    socket.onclose = () => {
      stopCapture();
      startButton.disabled = false;
      stopButton.disabled = true;
      if (statusText.textContent !== "Idle" && !statusDot.classList.contains("error")) {
        setStatus("Disconnected");
      }
    };

    await new Promise((resolve, reject) => {
      socket.addEventListener("open", resolve, { once: true });
      socket.addEventListener("error", reject, { once: true });
    });
    socket.onerror = () => setStatus("Connection error", "error");
  } catch (error) {
    stopSession();
    setStatus(error instanceof Error ? error.message : "Could not start", "error");
  }
}

startButton.addEventListener("click", startSession);
stopButton.addEventListener("click", stopSession);
clearButton.addEventListener("click", () => {
  finalLines = [];
  partialLine = "";
  renderTranscript();
});

async function loadServerConfig() {
  try {
    const response = await fetch("/config");
    if (!response.ok) {
      return;
    }

    const config = await response.json();
    modelInput.value = config.model || modelInput.value;
    realtimeModelInput.value = config.realtime_model || "";
    languageInput.value = config.language || languageInput.value;
  } catch {
    setStatus("Config unavailable");
  }
}

void loadServerConfig();

window.addEventListener("beforeunload", stopSession);
