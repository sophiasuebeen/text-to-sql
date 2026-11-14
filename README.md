# Text-to-SQL with QLoRA

Fine-tune Qwen2.5-Coder-1.5B-Instruct to generate SQLite queries from an English question and a schema. The goal is to measure adapter gains on held-out databases within a Colab T4 memory budget.

## Task

- Input:
    - Schema:   CREATE TABLE users (id INTEGER PRIMARY KEY, age INTEGER);
    - Question: How many users are at least 30 years old?
- Output:
    - SQL: SELECT COUNT(*) FROM users WHERE age >= 30;


## Training objective

- Let $x$ be the serialized schema and question, $y=(y_1,\ldots,y_T)$ the reference SQL followed by EOS, and $V$ the model vocabulary. Given the reference prefix $y_{\lt t}$, Qwen produces a hidden state $h_t$ and vocabulary logits $z_t=W_{\mathrm{head}}h_t$:

$$
p_{\theta_0,\phi}(y_t=k\mid x,y_{\lt t})
=\frac{\exp(z_{t,k})}{\sum_{j\in V}\exp(z_{t,j})}.
$$

- We freeze the base parameters $\theta_0$ and optimize the LoRA parameters $\phi$ using teacher-forced next-token cross-entropy:

$$
\min_{\phi}\;\mathcal{L}(\phi)
=-\frac{1}{N}\sum_i\sum_{t=1}^{T_i}
\log p_{\theta_0,\phi}(y_{i,t}\mid x_i,y_{i,\lt t}),
\qquad
N=\sum_i T_i
$$

- $N=$  the number of supervised tokens in an accumulation window, not the number of examples, so the loss is a per-token mean.

- SQL and EOS contribute to loss; EOS teaches stopping. Prompt and padding labels are `-100` (ignored by cross-entropy). Microbatches are right-padded to their longest sequence, with `attention_mask=0` on padding. Training sequences over 2,048 tokens are dropped and counted.


## Hyperparameters and Initialization

For each targeted linear transformation, with biases omitted:

$$
u=\widehat W_0 a+\frac{\alpha}{r}B\,A\,\mathrm{Dropout}(a),
\qquad
A\in\mathbb{R}^{r\times d_{\mathrm{in}}},\quad
B\in\mathbb{R}^{d_{\mathrm{out}}\times r}.
$$

- Here $\widehat W_0$ represents the NF4-quantized base weights. LoRA factorizes the learned update $\Delta W=(\alpha/r)BA$, not the pretrained $W_0$. Its output is added to the base transformation.

- Default PEFT initialization gives random $A$ and zero $B$, so the adapter initially adds zero. Backpropagation and AdamW update these matrices while the base weights remain frozen.


| Setting | This experiment |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-1.5B-Instruct`, pinned revision |
| Quantization | 4-bit NF4 with double quantization; FP16 compute on T4 |
| LoRA | `r=16`, `alpha=32` (scale 2), dropout 0.05 |
| Targets | `q_proj`, `k_proj`, `v_proj`, `o_proj` in attention blocks; no MLP adapters |
| Trainable parameters | 4,358,144 out of 1,548,072,448 (about 0.28%) |
| Optimizer | `paged_adamw_8bit`; initial learning rate `2e-4`, cosine decay, 3% warmup |
| Batching | 2 examples per microbatch, 8 gradient accumulation steps; effective batch 16 |

Adapter scope is attention-only rather than all-linear.

## flow
```text
main() → load_config() → train()
  ├─ build_model_and_tokenizer(): load Qwen → prepare k-bit training → attach LoRA
  ├─ prepare_training_data(): schema + question + SQL → tokens and masked labels
  ├─ collate(): pad examples and turn lists into tensors
  ├─ latest_complete_checkpoint(): validate and select resume state
  └─ Trainer.train()
       ├─ Qwen2ForCausalLM.forward(): hidden states → lm_head → logits
       ├─ ForCausalLMLoss(): align next-token targets and compute cross-entropy
       ├─ backward(): accumulate adapter gradients
       └─ optimizer.step(): update adapters after an accumulation window
```




| Question | Start here |
|---|---|
| What text does the model see? | [`build_messages`](src/text_to_sql/model/prompts.py), [`build_supervised_example`](src/text_to_sql/data/prepare.py) |
| Where are the model and adapters constructed? | [`build_model_and_tokenizer`](src/text_to_sql/model/train.py) |
| Who performs forward/backward and updates? | [`Trainer` setup](src/text_to_sql/model/train.py), [Qwen forward](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/qwen2/modeling_qwen2.py), [causal LM loss](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/loss/loss_utils.py) |
| How is interrupted training recovered? | [`checkpoints.py`](src/text_to_sql/model/checkpoints.py), callbacks in [`train.py`](src/text_to_sql/model/train.py) |
| How is SQL correctness determined? | [`score_prediction`](src/text_to_sql/evaluation/score.py), [`rows_match`](src/text_to_sql/evaluation/compare.py) |

## Saving and inference

:D


## Evaluation and current evidence

The primary metric is question-weighted **execution accuracy**:

$$
\mathrm{EX}=\frac{1}{M}\sum_{i=1}^{M}
\mathbf{1}\left[\mathrm{Exec}(\widehat q_i,D_i)
\equiv\mathrm{Exec}(q_i,D_i)\right].
$$

| Smoke evaluation | Eligible questions / DBs | Baseline | Adapter | Change |
|---|---:|---:|---:|---:|
| Fixed validation subset | 278 / 14 | 60.43% | 60.79% | +0.36 pp |
| Reconstructed training examples (diagnostic) | 184 / 85 | 54.89% | 64.13% | +9.24 pp |

These are smoke-run results. The held-out difference is one question (168 → 169 of 278), so it is not evidence of an adapter gain. Full-data evaluation is pending.



## Run locally or on Colab

**Mac / CI: Python 3.12+, no model weights or CUDA libraries required.**

```bash
uv sync --frozen
uv run pytest -q
```

**Drive a Colab session from the Mac:**

```bash
# Inspect an existing mounted session.
uv run python scripts/colab_training.py status --session sql-resume

# Resume an unfinished run after Drive has been mounted.
uv run python scripts/colab_training.py resume --session sql-resume --skip-mount --session-hours 2
```

**Inference on the Colab GPU**, from the installed project with Drive mounted:

```bash
python -m text_to_sql.model.predict \
  --adapter /content/drive/MyDrive/text-to-sql/runs/qlora-r16-lr2e-4-full-e1/final \
  --db /path/to/database.sqlite \
  --question 'How many users are at least 30 years old?' \
  --mode both --execute
```

## Further documentation

- [Evaluation rules and comparison commands](docs/evaluation.md)
- [Experiment records](docs/experiments.md), [data card](docs/data-card.md), and [model card](docs/model-card.md)
- [Architecture](docs/architecture.md) and [Colab command notebook](notebooks/colab-training-runbook.ipynb)
- [QLoRA paper and reference implementation](https://github.com/artidoro/qlora)
- [PEFT adapter checkpoint format](https://huggingface.co/docs/peft/developer_guides/checkpoint)

The base model is distributed under Apache-2.0; dataset and dependency terms apply separately. Weights and full databases are kept outside this repository.
