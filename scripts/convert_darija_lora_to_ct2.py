from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import torch
from ctranslate2.converters import TransformersConverter
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge the Darija Whisper LoRA adapter and convert it to CTranslate2."
    )
    parser.add_argument(
        "--base-model",
        default="openai/whisper-large-v3-turbo",
        help="Base Whisper model id or local path.",
    )
    parser.add_argument(
        "--adapter",
        default="anaszil/whisper-large-v3-turbo-darija",
        help="PEFT/LoRA adapter id or local path.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/darija",
        help="Output CTranslate2 directory. Mount this to /models/darija in Docker.",
    )
    parser.add_argument(
        "--merged-dir",
        default=None,
        help="Optional directory to keep the merged Transformers model.",
    )
    parser.add_argument(
        "--quantization",
        default="float16",
        help="CTranslate2 quantization. Use float16 for CUDA.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device used while merging the LoRA adapter.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output directories if they already exist.",
    )
    return parser.parse_args()


def select_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    merged_dir = Path(args.merged_dir).expanduser().resolve() if args.merged_dir else None

    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.force:
            raise SystemExit(f"{output_dir} is not empty. Pass --force to overwrite it.")
        shutil.rmtree(output_dir)

    device = select_device(args.device)
    dtype = torch.float16 if device == "cuda" else torch.float32

    with tempfile.TemporaryDirectory(prefix="darija-merged-") as tmpdir:
        transient_merged_dir = Path(tmpdir) / "merged"
        save_dir = merged_dir or transient_merged_dir
        if save_dir.exists() and any(save_dir.iterdir()):
            if not args.force:
                raise SystemExit(f"{save_dir} is not empty. Pass --force to overwrite it.")
            shutil.rmtree(save_dir)

        print(f"Loading base model: {args.base_model}", flush=True)
        base_model = WhisperForConditionalGeneration.from_pretrained(
            args.base_model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        base_model.to(device)

        print(f"Loading and merging adapter: {args.adapter}", flush=True)
        peft_model = PeftModel.from_pretrained(base_model, args.adapter)
        merged_model = peft_model.merge_and_unload()
        merged_model.eval()

        print(f"Saving merged Transformers model to: {save_dir}", flush=True)
        save_dir.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(save_dir, safe_serialization=True)

        try:
            processor = WhisperProcessor.from_pretrained(args.adapter)
        except Exception:
            processor = WhisperProcessor.from_pretrained(args.base_model)
        processor.save_pretrained(save_dir)

        print(f"Converting to CTranslate2 at: {output_dir}", flush=True)
        TransformersConverter(str(save_dir)).convert(
            str(output_dir),
            quantization=args.quantization,
            force=True,
        )

    model_bin = output_dir / "model.bin"
    if not model_bin.is_file():
        raise SystemExit(f"Conversion did not create {model_bin}.")

    print(f"Done. Mount {output_dir} to /models/darija.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
