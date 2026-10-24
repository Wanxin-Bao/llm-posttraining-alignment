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


## Day 2 — Preference Data and Reward Modeling

### Objective

The second stage of the project trains a reward model to distinguish preferred assistant responses from less-preferred alternatives.

Unlike supervised fine-tuning, which provides token-level target responses, reward modeling uses relative preference supervision:

\[
(x, y_{\text{chosen}}, y_{\text{rejected}})
\]

and learns a scalar reward function such that:

\[
r_\theta(x,y_{\text{chosen}})
>
r_\theta(x,y_{\text{rejected}})
\]

The reward model is initialized from the Day-1 Qwen3-0.6B SFT checkpoint. The SFT-trained Transformer backbone is reused, while the language-modeling head is replaced by a scalar sequence-classification reward head.

### Preference Dataset

The experiment uses `HuggingFaceH4/ultrafeedback_binarized`.

The original preference splits contain:

| Split | Preference pairs |
|---|---:|
| train_prefs | 61,135 |
| test_prefs | 2,000 |

A preference-quality audit was performed before training.

In the training split:

| Preference margin | Count | Fraction |
|---|---:|---:|
| score_chosen > score_rejected | 53,748 | 87.92% |
| score_chosen = score_rejected | 7,387 | 12.08% |
| score_chosen < score_rejected | 0 | 0% |

Pairs with equal chosen and rejected scores were removed so that the reward-model objective only used examples containing an explicit positive preference.

### Sequence-Length Analysis

A sample of 2,000 preference pairs was tokenized using the Qwen3 chat template.

| Statistic | Tokens |
|---|---:|
| Minimum | 28 |
| Median | 363 |
| P90 | 824 |
| P95 | 988 |
| P99 | 1,440 |
| Maximum | 2,459 |

A maximum sequence length of 1,024 tokens was selected. This covers approximately 95% of the analyzed responses while reducing the compute and memory cost of processing both chosen and rejected sequences.

### Reward-Model Objective

For each preference pair, the model produces two scalar rewards:

\[
r_c=r_\theta(x,y_{\text{chosen}})
\]

and

\[
r_r=r_\theta(x,y_{\text{rejected}})
\]

The reward model is trained using the pairwise Bradley–Terry objective:

\[
\mathcal{L}_{RM}
=
-\log \sigma(r_c-r_r)
\]

The objective encourages preferred responses to receive higher reward than rejected responses.

The absolute reward value is not interpreted as an objective quality score. The meaningful quantity is the relative ordering and reward margin between responses.

### Training Configuration

| Setting | Value |
|---|---|
| Initialization | Qwen3-0.6B-SFT |
| Hardware | NVIDIA GeForce RTX 3090 |
| Fine-tuning | Full-parameter |
| Precision | BF16 |
| Max sequence length | 1,024 |
| Per-device batch size | 1 preference pair |
| Gradient accumulation | 8 |
| Effective batch size | 8 pairs |
| Learning rate | 1e-5 |
| Optimizer | AdamW fused |
| Scheduler | Cosine |
| Epochs | 1 |

After preference-score filtering and sequence-length filtering, approximately 50.6k training pairs were used.

### Smoke-Test Result

Before reward-model training, the newly initialized reward head ranked held-out preference pairs at approximately chance level:

| Metric | Before RM training | After 20 steps |
|---|---:|---:|
| Pairwise accuracy | 47.74% | 59.67% |
| Eval loss | 0.7735 | 0.6693 |
| Mean reward margin | 0.036 | 0.227 |

The rapid increase in held-out ranking accuracy confirmed that the preference-data pipeline, scalar reward head, pairwise objective, and backbone updates were functioning correctly.

### Full Training Results

The final reward model was trained for one epoch.

| Metric | Result |
|---|---:|
| Optimizer steps | 6,328 |
| Training runtime | ~2 h 28 min |
| Train loss | 0.5075 |
| Train pairwise accuracy | 73.77% |
| Train reward margin | 1.124 |
| Held-out pairs | 1,636 |
| Eval loss | 0.4721 |
| Eval pairwise accuracy | **76.89%** |
| Eval reward margin | **1.391** |

Pairwise accuracy improved from approximately 47.7% before preference training to 76.9% after one epoch, an absolute increase of approximately 29.2 percentage points.

### Qualitative Reward Analysis

The trained reward model successfully ranked many examples with clear differences in instruction following, relevance, completeness, and response quality.

However, failure analysis also exposed limitations of both the reward model and the preference data.

Some incorrectly ranked examples contained noisy or questionable preference labels. In one mathematical example, the dataset-preferred response contained factual inconsistencies, while the reward model assigned it a lower reward.

Other errors involved subtle preference trade-offs. For example, a task explicitly requesting a `True` or `False` label contained a dataset preference for a longer answer with an explanation, while the reward model preferred the shorter response that followed the output format more strictly.

These examples illustrate an important limitation of reward modeling:

**A reward model learns the preference distribution represented by its training data; it is not an objective ground-truth estimator of response quality.**

This creates potential failure modes including preference bias, reward-model misgeneralization, and reward overoptimization.

### Day 2 Summary

Day 1 used token-level supervised learning to teach the model how an assistant should respond.

Day 2 changed the supervision signal from absolute target responses to relative preference pairs:

**SFT learns how to answer; reward modeling learns which answer should be preferred.**

The trained reward model will later provide the explicit reward signal required for the PPO stage of the project. In contrast, DPO will demonstrate how preference optimization can update the policy directly without training or querying a separate reward model during optimization.