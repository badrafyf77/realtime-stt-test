from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import torch
from ctranslate2.converters import TransformersConverter
from peft import PeftModel
from tokenizers import Tokenizer
from transformers import AutoTokenizer, WhisperFeatureExtractor, WhisperForConditionalGeneration


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
        "--tokenizer-source",
        default=None,
        help="Tokenizer/feature extractor id or path. Defaults to --base-model.",
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


def save_processor_files(source: str, save_dir: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(f"{source} did not load a fast tokenizer.")
    tokenizer.save_pretrained(save_dir)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(source)
    feature_extractor.save_pretrained(save_dir)

    expected_files = ["tokenizer.json", "preprocessor_config.json"]
    missing_files = [filename for filename in expected_files if not (save_dir / filename).is_file()]
    if missing_files:
        raise RuntimeError(
            "Saving tokenizer/feature extractor did not create required files: "
            + ", ".join(missing_files)
        )

    raw_tokenizer = Tokenizer.from_file(str(save_dir / "tokenizer.json"))
    for sample in [" -", " '"]:
        token_ids = raw_tokenizer.encode(sample).ids
        if not token_ids:
            raise RuntimeError(
                f"Saved tokenizer.json cannot encode {sample!r}; faster-whisper will fail."
            )


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

        tokenizer_source = args.tokenizer_source or args.base_model
        print(f"Saving tokenizer and feature extractor from: {tokenizer_source}", flush=True)
        save_processor_files(tokenizer_source, save_dir)

        print(f"Converting to CTranslate2 at: {output_dir}", flush=True)
        TransformersConverter(
            str(save_dir),
            copy_files=["tokenizer.json", "preprocessor_config.json"],
        ).convert(
            str(output_dir),
            quantization=args.quantization,
            force=True,
        )

    required_files = [
        "model.bin",
        "config.json",
        "tokenizer.json",
        "preprocessor_config.json",
        "vocabulary.json",
    ]
    missing_files = [filename for filename in required_files if not (output_dir / filename).is_file()]
    if missing_files:
        raise SystemExit(
            "Conversion finished, but the CT2 directory is incomplete. "
            f"Missing: {', '.join(missing_files)}"
        )

    print(f"Done. Mount {output_dir} to /models/darija.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
