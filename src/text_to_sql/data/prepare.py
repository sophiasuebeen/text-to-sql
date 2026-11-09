"""Turn examples into supervised training features (tokenization, label masking, padding).

Tokenizer functions accept any object with the Hugging Face tokenizer interface:
on the Mac they are tested with a fake tokenizer; in Colab the pinned real
tokenizer must be re-validated, since a fake cannot prove the chat template.
"""

from __future__ import annotations

from typing import Sequence

from ..model.prompts import build_messages

IGNORE_INDEX = -100


class TemplateContractError(ValueError):
    """The tokenizer/chat template does not satisfy the frozen prompt contract."""


def prompt_text(tokenizer, schema: str, question: str) -> str:
    return tokenizer.apply_chat_template(build_messages(schema, question), tokenize=False, add_generation_prompt=True)


def build_supervised_example(tokenizer, schema: str, question: str, sql: str) -> dict[str, list[int]]:
    """Token IDs with labels masked to -100 everywhere except the SQL and EOS.

    The boundary is found from character offsets of the *full* tokenized string,
    never by concatenating separately tokenized pieces.
    """
    prompt = prompt_text(tokenizer, schema, question)
    target = sql.strip() + tokenizer.eos_token
    full = prompt + target
    rendered = tokenizer.apply_chat_template(build_messages(schema, question, sql), tokenize=False)
    if not rendered.startswith(full):
        raise TemplateContractError("chat template does not render prompt + SQL + EOS as a prefix")

    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = list(enc["input_ids"]), enc["offset_mapping"]
    boundary = len(prompt)
    labels = []
    for tid, (start, end) in zip(ids, offsets):
        if start < boundary < end:
            raise TemplateContractError("a token straddles the prompt/answer boundary")
        labels.append(tid if start >= boundary else IGNORE_INDEX)
    if not ids or ids[-1] != tokenizer.eos_token_id or labels[-1] != tokenizer.eos_token_id:
        raise TemplateContractError("sequence must end with a supervised EOS token")
    if all(lab == IGNORE_INDEX for lab in labels[:-1]):
        raise TemplateContractError("no SQL tokens are supervised")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def pad_batch(features: Sequence[dict[str, list[int]]], pad_token_id: int) -> dict[str, list[list[int]]]:
    """Right-pad. Padding is masked by position, so EOS stays supervised even if pad == eos."""
    width = max(len(f["input_ids"]) for f in features)
    out: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for f in features:
        pad = width - len(f["input_ids"])
        out["input_ids"].append(f["input_ids"] + [pad_token_id] * pad)
        out["attention_mask"].append(f["attention_mask"] + [0] * pad)
        out["labels"].append(f["labels"] + [IGNORE_INDEX] * pad)
    return out


def prompt_fits(n_prompt_tokens: int, max_context: int, max_new_tokens: int) -> bool:
    """Dev/test prompts that do not fit are model failures, never truncated."""
    return n_prompt_tokens + max_new_tokens <= max_context
