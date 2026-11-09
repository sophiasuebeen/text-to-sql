"""Shared test doubles."""

import re


class FakeTokenizer:
    """Word-ish tokenizer with ChatML-like template; exercises orchestration only."""

    eos_token = "<|im_end|>"
    eos_token_id = 2
    pad_token_id = 2

    def __init__(self, render_eos=True):
        self.render_eos = render_eos
        self.vocab = {"<|im_end|>": 2}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        s = ""
        for m in messages:
            end = self.eos_token if self.render_eos else "<|end|>"
            s += f"<|im_start|>{m['role']}\n{m['content']}{end}\n"
        if add_generation_prompt:
            s += "<|im_start|>assistant\n"
        return s

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids, offsets = [], []
        for m in re.finditer(r"<\|[a-z_]+\|>|\w+|[^\w\s]|\s+", text):
            tok = m.group()
            ids.append(self.vocab.setdefault(tok, len(self.vocab) + 10))
            offsets.append((m.start(), m.end()))
        return {"input_ids": ids, "offset_mapping": offsets}
