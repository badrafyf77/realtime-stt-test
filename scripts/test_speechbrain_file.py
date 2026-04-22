from __future__ import annotations

import argparse
import os
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe one audio file with a SpeechBrain ASR model.")
    parser.add_argument("audio_file", help="Path to a WAV/audio file. 16 kHz mono WAV is preferred.")
    parser.add_argument("--model", default="aioxlabs/dvoice-darija", help="SpeechBrain model id.")
    parser.add_argument(
        "--savedir",
        default=os.getenv("SPEECHBRAIN_SAVEDIR", "pretrained_models/asr-wav2vec2-dvoice-dar"),
        help="Local SpeechBrain model cache directory.",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("STT_DEVICE", "cuda"),
        choices=["cuda", "cpu"],
        help="Device passed as SpeechBrain run_opts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from speechbrain.inference.ASR import EncoderASR
    except ImportError:
        from speechbrain.pretrained import EncoderASR

    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    except Exception as exc:
        print(f"Could not inspect torch CUDA status: {exc}", flush=True)

    print(f"Loading model={args.model} savedir={args.savedir} device={args.device}", flush=True)
    started_at = time.perf_counter()
    asr_model = EncoderASR.from_hparams(
        source=args.model,
        savedir=args.savedir,
        run_opts={"device": args.device},
    )
    print(f"Loaded in {time.perf_counter() - started_at:.3f}s", flush=True)

    started_at = time.perf_counter()
    text = asr_model.transcribe_file(args.audio_file)
    print(f"Inference time: {time.perf_counter() - started_at:.3f}s", flush=True)
    print("Transcript:")
    print(text)


if __name__ == "__main__":
    main()
