## Day 1 — Supervised Fine-Tuning Baseline

### Objective

The first stage of the project establishes a supervised fine-tuning (SFT) baseline by converting a pretrained language model into an instruction-following assistant.

The experiment uses `Qwen/Qwen3-0.6B-Base`, a pretrained base model, and fine-tunes all model parameters on the conversational `trl-lib/Capybara` dataset.

The goal of this stage is not to maximize final response quality, but to establish a reproducible post-training baseline and measure the behavioral change introduced by SFT.

### Dataset

`trl-lib/Capybara` contains:

- 15,806 training conversations
- 200 evaluation conversations
- Multi-turn `user` / `assistant` message format

A sample of 1,000 training conversations was analyzed to determine sequence lengths:

| Statistic | Tokens |
|---|---:|
| Minimum | 31 |
| Median | 878 |
| P90 | 1,863 |
| P95 | 2,266 |
| Maximum | 3,882 |

Based on this distribution, a maximum sequence length of 2,048 tokens was selected as a trade-off between context coverage, GPU memory usage, and training throughput.

### Training Configuration

| Setting | Value |
|---|---|
| Model | Qwen3-0.6B-Base |
| Hardware | NVIDIA GeForce RTX 3090 |
| Precision | BF16 |
| Fine-tuning | Full-parameter |
| Max sequence length | 2,048 |
| Per-device batch size | 1 |
| Gradient accumulation | 8 |
| Effective batch size | 8 |
| Optimizer | AdamW fused |
| Learning rate | 2e-5 |
| LR scheduler | Cosine |
| Warmup steps | 60 |
| Gradient checkpointing | Enabled |
| Loss | Assistant tokens only |
| Epochs | 1 |

A training-compatible Qwen3 chat template was used so that the loss is applied only to assistant-generated tokens rather than user messages.

### Training Results

The model was trained for one full epoch over 15,806 conversations.

| Metric | Result |
|---|---:|
| Optimizer steps | 1,969 |
| Training runtime | 2,845 s |
| Training runtime | ~47.4 min |
| Training throughput | 5.535 samples/s |
| Optimizer throughput | 0.692 steps/s |
| Train loss | 1.133 |
| Eval loss | 0.9886 |
| Eval mean token accuracy | 0.7381 |
| Final model size | ~1.2 GB |

Training remained stable throughout the run, with no NaN losses, gradient divergence, or GPU out-of-memory failures.

### Base vs. SFT Behavioral Comparison

Both models were evaluated using the same five prompts, chat-template settings, greedy decoding, and generation length. Thinking mode was disabled to make the comparison easier to interpret.

| Task | Base Model | SFT Model |
|---|---|---|
| Explain why the sky is blue | Mostly invalid/special-character output | Produces a coherent Rayleigh-scattering explanation, although with repetition and a factual error |
| Write a Python max function | Invalid repetitive output | Produces a valid implementation including empty-list handling |
| Technical interview tips | Partially coherent but malformed/truncated | Produces three structured recommendations |
| Supervised vs. reinforcement learning | Invalid/special-character output | Produces a coherent conceptual comparison |
| Email requesting a professor meeting | Invalid/special-character output | Produces a structured professional email |

The most significant effect of SFT is therefore behavioral rather than purely knowledge-based.

Before fine-tuning, the pretrained model behaves primarily as a next-token continuation model and does not reliably follow conversational instructions. After SFT, it consistently adopts an assistant role and follows multiple instruction formats.

However, SFT does not guarantee high response quality. The resulting model still exhibits repetition, generic wording, factual mistakes, and suboptimal phrasing.

This motivates the next stages of the project:

**SFT teaches the model what an assistant response should look like; preference optimization aims to teach the model which assistant response is better.**
