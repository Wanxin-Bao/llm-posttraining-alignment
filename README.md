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

## Day 3 — Direct Preference Optimization

### Objective

Day 3 applies **Direct Preference Optimization (DPO)** to the supervised fine-tuned Qwen3-0.6B policy from Day 1.

Unlike the classical RLHF pipeline used for reward modeling and PPO, DPO does not require an explicit reward model during policy optimization. Instead, it directly learns from preference pairs consisting of a preferred (`chosen`) and dispreferred (`rejected`) response.

The trainable policy and frozen reference policy are both initialized from the Day 1 SFT checkpoint:

```text
Qwen3-0.6B-SFT
      │
      ├── Trainable policy πθ
      │
      └── Frozen reference policy πref
                │
                ▼
       chosen / rejected pairs
                │
                ▼
              DPO
```

### DPO Objective

For a prompt \(x\), preferred response \(y_c\), and rejected response \(y_r\), the DPO loss is

\[
\mathcal{L}_{DPO}
=
-\log \sigma
\left(
\beta
\left[
\log \frac{\pi_\theta(y_c|x)}
{\pi_{\mathrm{ref}}(y_c|x)}
-
\log \frac{\pi_\theta(y_r|x)}
{\pi_{\mathrm{ref}}(y_r|x)}
\right]
\right).
\]

The frozen SFT reference model acts as an anchor, allowing preference optimization without allowing the policy to drift arbitrarily far from the original instruction-following behavior.

At initialization,

\[
\pi_\theta = \pi_{\mathrm{ref}},
\]

so both implicit rewards are zero and the theoretical initial loss is

\[
-\log \sigma(0)
=
0.6931.
\]

This was reproduced by the implementation during the pre-DPO evaluation.

### Preference Dataset

The same preference source used for Day 2 reward modeling was used:

```text
HuggingFaceH4/ultrafeedback_binarized
```

Pairs with equal preference scores were removed:

```text
score_chosen > score_rejected
```

This retained:

```text
Training preference pairs:   53,748
Evaluation preference pairs:  1,728
```

The DPO representation explicitly separates the common prompt from the two assistant completions:

```text
prompt
├── chosen assistant response
└── rejected assistant response
```

A maximum sequence length of 1024 tokens was retained from the Day 2 dataset analysis.

### Training Configuration

```text
Model initialization       Qwen3-0.6B-SFT
Training method            Full-parameter DPO
Epochs                     1
Beta                       0.1
Learning rate              1e-6
Max sequence length        1024
Precision                  BF16
Per-device batch size      1
Gradient accumulation      8
Effective batch size       8
Optimizer                  AdamW fused
Gradient checkpointing     Enabled
Hardware                   NVIDIA RTX 3090
```

A small learning-rate study was performed before the full run.

| Configuration | Eval Loss | Preference Accuracy | Reward Margin |
|---|---:|---:|---:|
| 200 steps, LR \(5\times10^{-7}\) | 0.6940 | 46.53% | 0.0008 |
| 200 steps, LR \(1\times10^{-6}\) | 0.6924 | 51.29% | 0.0043 |
| 1000 steps, LR \(1\times10^{-6}\) | 0.6794 | 59.38% | 0.0330 |

The larger learning rate produced a clearer optimization signal and was therefore selected for the full one-epoch run.

### Full DPO Results

The final training run completed one full epoch in approximately 3.7 hours.

```text
Training loss:              0.6523
Evaluation loss:            0.6493
Held-out preference acc.:   65.73%
Implicit reward margin:     0.1173

Chosen implicit reward:    -0.0909
Rejected implicit reward:  -0.2082
```

The evaluation loss decreased from the neutral DPO initialization of approximately 0.6931 to 0.6493, while held-out preference accuracy reached 65.73%.

Both chosen and rejected implicit rewards became negative. However, the rejected responses were suppressed substantially more strongly:

\[
-0.0909 > -0.2082.
\]

Consequently,

\[
r_{\mathrm{chosen}} - r_{\mathrm{rejected}}
=
0.1173 > 0.
\]

This shows that preference separation emerged primarily by decreasing the relative likelihood of rejected responses more strongly than that of chosen responses, rather than simply increasing the absolute probability of preferred responses.

### SFT vs. DPO Generation Evaluation

DPO training metrics alone do not establish that free-generated responses are better than those produced by the original SFT policy.

To evaluate actual generation behavior, both models generated responses for 1,000 held-out prompts using deterministic decoding. The separately trained Day 2 reward model then scored each SFT and DPO response.

```text
Prompts evaluated:       1000

DPO preferred:            404 (40.40%)
SFT preferred:            322 (32.20%)
Ties:                     274 (27.40%)

Mean SFT reward:          0.4366
Mean DPO reward:          0.5298
Mean reward difference:  +0.0932
```

Among examples for which the reward model expressed a strict preference,

\[
\text{DPO win rate}
=
\frac{404}{404+322}
=
55.65\%.
\]

### Response Preservation Analysis

The relatively large number of reward ties was investigated further.

Of the 274 ties:

```text
Identical SFT/DPO responses: 260
Different responses:           14
```

Therefore, 94.89% of reward ties occurred because the SFT and DPO policies generated exactly the same response.

Across all 1,000 prompts:

```text
Exact same response:   260 (26.0%)
Changed response:      740 (74.0%)
```

Among changed responses:

```text
DPO wins:              404
SFT wins:              322
Reward-model ties:      14
```

Excluding the 14 unresolved ties, the DPO policy achieved a 55.65% win rate among cases in which post-training actually changed the generated response.

This suggests that DPO produced a moderate positive preference shift while retaining part of the original SFT behavior.

### Interpretation

The Day 3 experiments demonstrate two complementary effects.

First, the DPO policy successfully learned the pairwise preference objective, reaching 65.73% held-out preference accuracy and increasing the implicit reward margin from zero to 0.1173.

Second, this optimization transferred to free generation. A separately trained reward model preferred DPO generations more frequently than SFT generations, with a 55.65% win rate when ties were excluded.

The improvement was moderate rather than universal: approximately one quarter of evaluated prompts produced exactly the same response before and after DPO, and SFT remained preferred on a substantial fraction of changed generations.

### Limitation

The Day 2 reward model and Day 3 DPO policy are separately trained models, but both learn from the UltraFeedback preference distribution.

Therefore, the reward-model evaluation should be interpreted as evidence of improved alignment to the learned UltraFeedback preference distribution rather than as an independent measurement of human preference or absolute response quality.