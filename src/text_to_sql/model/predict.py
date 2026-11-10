"""Load a saved LoRA export and generate SQL, with the adapter on or off.

ML dependencies are loaded only when constructing the predictor on a GPU.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

from ..data.prepare import prompt_fits, prompt_text
from ..data.schema import SERIALIZER_VERSION, serialize_schema
from ..sql.executor import execute_sql
from ..sql.policy import extract_sql
from .checkpoints import is_complete, verify_manifest
from .prompts import PROMPT_VERSION


@dataclass(frozen=True)
class Prediction:
    """elapsed_s is synchronized generation time; zero when generation is skipped."""

    raw: str
    sql: str | None
    status: str
    prompt_tokens: int
    elapsed_s: float


def export_identity(adapter: str | Path) -> dict:
    """Reject incomplete, corrupted, or prompt-incompatible exports before loading."""
    adapter = Path(adapter)
    if not is_complete(adapter):
        raise ValueError(f"export is not marked complete: {adapter}")
    verify_manifest(adapter)
    identity = json.loads((adapter / "identity.json").read_text())
    marker = json.loads((adapter / "COMPLETE.json").read_text())
    if marker["identity"] != identity:
        raise ValueError("export identity disagrees with completion marker")
    if not identity.get("model_revision"):
        raise ValueError("export has no pinned base-model revision")
    if (identity["prompt_version"], identity["serializer_version"]) != (PROMPT_VERSION, SERIALIZER_VERSION):
        raise ValueError("export prompt/schema versions differ from the current code")
    return identity


class SQLPredictor:
    def __init__(self, adapter: str | Path, *, max_context: int = 2048, max_new_tokens: int = 256):
        if not 0 < max_new_tokens < max_context:
            raise ValueError("require 0 < max_new_tokens < max_context")
        self.identity = export_identity(adapter)
        self.max_context = max_context
        self.max_new_tokens = max_new_tokens

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("this 4-bit inference path requires a CUDA GPU")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(str(adapter))
        base = AutoModelForCausalLM.from_pretrained(
            self.identity["model"], revision=self.identity["model_revision"],
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
            ),
            torch_dtype=dtype, device_map={"": 0},
        )
        self.model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
        self.model.eval()

    def predict(self, schema: str, question: str, *, baseline: bool = False) -> Prediction:
        import torch

        encoded = self.tokenizer(
            prompt_text(self.tokenizer, schema, question),
            add_special_tokens=False, return_tensors="pt",
        )
        n = encoded["input_ids"].shape[-1]
        if not prompt_fits(n, self.max_context, self.max_new_tokens):
            return Prediction("", None, "prompt_overflow", n, 0.0)
        encoded = encoded.to(self.model.device)
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        # The same quantized base and tokenizer are used for both conditions.
        context = self.model.disable_adapter() if baseline else nullcontext()
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.inference_mode(), context:
            output = self.model.generate(
                **encoded, do_sample=False, num_beams=1, use_cache=True,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=pad, eos_token_id=self.tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        tokens = output[0, n:].tolist()
        raw = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        if len(tokens) >= self.max_new_tokens and tokens[-1] != self.tokenizer.eos_token_id:
            return Prediction(raw, None, "generation_limit", n, elapsed)
        extracted = extract_sql(raw)
        return Prediction(raw, extracted.sql, "ok" if extracted.ok else extracted.error, n, elapsed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--question", required=True)
    ap.add_argument("--mode", choices=["baseline", "adapter", "both"], default="both")
    ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--execute", action="store_true", help="execute validated SQL in the read-only sandbox")
    args = ap.parse_args()
    schema = serialize_schema(args.db)
    predictor = SQLPredictor(args.adapter, max_context=args.max_context, max_new_tokens=args.max_new_tokens)
    modes = ["baseline", "adapter"] if args.mode == "both" else [args.mode]
    for mode in modes:
        prediction = predictor.predict(schema, args.question, baseline=mode == "baseline")
        result = {"mode": mode, **asdict(prediction)}
        if args.execute and prediction.sql is not None:
            outcome = execute_sql(args.db, prediction.sql)
            result["execution"] = {
                "status": outcome.status, "error": outcome.error, "ncols": outcome.ncols,
                "row_count": len(outcome.rows), "canonical_rows_preview": outcome.to_json()["rows"][:20],
            }
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
