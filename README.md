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

## Day 4 — PPO: Online RLHF with Reward and Value Models

### Goal

Day 4 implements **Proximal Policy Optimization (PPO)** as a classical online RLHF baseline.

Unlike SFT, reward modeling, and DPO, which train on fixed offline responses, PPO performs **online rollouts**:

1. Sample prompts from a prompt-only dataset.
2. Generate responses using the current policy.
3. Score generated responses with a frozen reward model.
4. Apply a KL penalty relative to the frozen SFT reference policy.
5. Estimate advantages using a trainable value model and GAE.
6. Update the policy using the PPO clipped objective.
7. Update the value model to predict future returns.

The main comparison is therefore:

```text
SFT → DPO
vs.
SFT → PPO
```

Both methods start from the same Day 1 SFT checkpoint.

---

### Model Components

The PPO pipeline uses four model roles:

```text
Policy
  └── Day 1 SFT checkpoint
      trainable

Reference Policy
  └── Day 1 SFT checkpoint
      frozen

Reward Model
  └── Day 2 reward-model checkpoint
      frozen

Value Model / Critic
  └── initialized from Day 2 reward-model checkpoint
      trainable
```

The policy and reference policy start from identical SFT weights.

The reward model evaluates the complete generated response, while the value model learns to predict expected future return from intermediate generation states.

---

### PPO Objective

For a sampled action \(a_t\) at state \(s_t\), the KL-related log-ratio relative to the frozen reference policy is

\[
k_t =
\log \pi_\theta(a_t|s_t)
-
\log \pi_{\text{ref}}(a_t|s_t).
\]

The token-level KL reward is

\[
r_t^{\mathrm{KL}}
=
-\beta_{\mathrm{KL}} k_t.
\]

The reward-model score is added to the final valid response token.

Advantages are estimated using Generalized Advantage Estimation:

\[
\delta_t
=
r_t
+
\gamma V(s_{t+1})
-
V(s_t),
\]

\[
A_t
=
\delta_t
+
\gamma\lambda A_{t+1}.
\]

The PPO policy update uses

\[
r_t(\theta)
=
\frac{
\pi_\theta(a_t|s_t)
}{
\pi_{\mathrm{old}}(a_t|s_t)
},
\]

and the clipped objective

\[
L_{\mathrm{PPO}}
=
-
\mathbb{E}
\left[
\min
\left(
r_t(\theta)A_t,
\operatorname{clip}
(
r_t(\theta),
1-\epsilon,
1+\epsilon
)
A_t
\right)
\right].
\]

The frozen SFT reference policy and the PPO old policy have different roles:

- **Reference policy:** controls long-term drift through the KL penalty.
- **Old policy:** controls the size of each individual PPO update.

---

### Prompt-Only Dataset

PPO requires prompts rather than fixed chosen/rejected response pairs.

The prompt dataset was derived from the same UltraFeedback preference data used in Days 2–3.

Preparation:

```text
Raw train_prefs:             61,135
Positive-margin pairs:       53,748
After prompt deduplication:  53,737
Prompt length ≤ 512:         49,897

PPO train prompts:           49,797
Internal eval prompts:          100
```

Prompts longer than 512 tokens were dropped rather than truncated to avoid corrupting instructions.

The final dataset contains only:

```text
input_ids
```

and is stored at:

```text
data/ppo_prompts_512
```

---

### Manual PPO Diagnostics

Before running PPO training, several small diagnostic scripts were used to inspect the full RLHF pipeline.

#### Online rollout

The SFT policy generated responses online and the frozen reward model scored them.

One rollout revealed an important limitation of learned reward models: a response that failed the requested output format still received a positive reward.

This demonstrates a practical RLHF concern:

```text
Reward model score ≠ guaranteed response quality
```

and motivates monitoring for reward-model errors and reward hacking.

#### KL verification

At initialization,

```text
Policy = Reference Policy
```

therefore

\[
\log
\frac{\pi_\theta}{\pi_{\text{ref}}}
=
0.
\]

The manual test confirmed zero sampled KL at initialization.

#### Value and GAE inspection

The value model was initialized from the reward-model checkpoint.

Because the reward model was trained to score complete responses rather than partial generation states, its initial value estimates were noisy.

This is expected: during PPO, the critic must learn a different task—

> predicting expected future return from partial generation states.

---

### Rollout-Length Ablation

Initial rollouts frequently reached the maximum response length without generating EOS.

A controlled ablation compared:

```text
response_length = 128
response_length = 256
```

All other PPO settings were held constant.

| Metric | 128 tokens | 256 tokens |
|---|---:|---:|
| Episodes | 64 | 64 |
| EOS rate | 46.88% | **67.19%** |
| Mean RM score | -1.8174 | **-1.4940** |
| Mean RLHF reward | -1.8329 | **-1.5029** |
| Mean sampled KL | 0.3107 | 0.1777 |
| PPO approx KL | 0.000995 | 0.000585 |
| Value loss | 1.5359 | 1.5844 |
| Policy ratio | 0.999442 | 0.999872 |
| Clip fraction | 0.66% | 2.02% |
| Peak VRAM | 5524 MiB | 5610 MiB |

Increasing the rollout budget from 128 to 256 tokens improved the EOS completion rate by more than 20 percentage points while increasing peak GPU memory by only approximately 86 MiB.

Therefore:

```text
response_length = 256
```

was selected for the main PPO experiment.

---

### Main PPO Configuration

```text
Policy initialization        Day 1 SFT
Reference policy             frozen Day 1 SFT
Reward model                 frozen Day 2 RM
Value model                  Day 2 RM initialization

Prompt length                ≤ 512
Response length              256

Total episodes               4096
Learning rate                1e-6
Temperature                  0.7

Per-device train batch       1
Gradient accumulation        4
Rollout batch                4

PPO epochs                   1
Mini-batches                 1

KL coefficient               0.05
Policy clip range            0.2
Value clip range             0.2
Value-loss coefficient       0.1

Gamma                        1.0
GAE lambda                   0.95

Missing-EOS penalty          1.0

Precision                    BF16
Training                     full-parameter
```

The final policy is saved to:

```text
outputs/qwen3-0.6b-ppo
```

---

### Training Results

The 4096-episode PPO run completed successfully.

```text
PPO updates:   1024
Episodes:      4096
Runtime:       ~3 h 39 min
```

Overall statistics:

| Metric | Value |
|---|---:|
| Mean RM score | -1.6006 |
| Mean RLHF reward | -1.6169 |
| Mean sampled KL | 0.3267 |
| PPO approx KL | 0.000705 |
| Policy ratio | 0.999939 |
| Clip fraction | 0.69% |
| Value loss | 1.2342 |
| EOS rate | 67.04% |

The training process remained numerically stable:

```text
No NaNs
No CUDA OOM
Policy ratio ≈ 1
Very low PPO clipping fraction
Small per-update approximate KL
```

---

### Critic Learning

The value loss decreased consistently across training:

| Training quarter | Value loss |
|---|---:|
| Q1 | 1.4248 |
| Q2 | 1.2450 |
| Q3 | 1.1522 |
| Q4 | **1.1146** |

This indicates that the critic progressively learned to estimate PPO returns from partial generation states.

---

### Policy Drift

The sampled KL relative to the frozen SFT reference increased during training:

| Training quarter | Sampled KL |
|---|---:|
| Q1 | 0.2492 |
| Q2 | 0.2960 |
| Q3 | 0.3356 |
| Q4 | **0.4261** |

At the same time, the PPO per-update ratio remained approximately 1.

This indicates a useful distinction:

```text
Each individual PPO update was small,
but many small updates accumulated into measurable drift from the SFT policy.
```

The EOS rate remained approximately constant:

```text
Q1: 67.29%
Q2: 67.38%
Q3: 66.02%
Q4: 67.48%
```

Therefore the missing-EOS penalty did not produce a measurable improvement in completion behavior during this run.

---

### Held-Out Evaluation: SFT vs PPO

The final PPO policy was evaluated on 1000 held-out prompts using the same protocol as the Day 3 DPO evaluation.

For each prompt:

```text
SFT → greedy response
PPO → greedy response

Frozen Day 2 RM → score both responses
```

Results:

| Metric | Result |
|---|---:|
| PPO wins | 303 / 1000 |
| SFT wins | 304 / 1000 |
| Ties | 393 / 1000 |
| Mean SFT reward | 0.436594 |
| Mean PPO reward | 0.427900 |
| Mean reward difference | **-0.008694** |
| PPO non-tie win rate | **49.92%** |

The PPO checkpoint therefore showed **no measurable preference improvement over the SFT baseline** under this evaluation protocol.

---

### Response-Change Analysis

The PPO policy nevertheless changed a substantial fraction of greedy outputs:

```text
Identical responses: 381 / 1000 = 38.10%
Changed responses:   619 / 1000 = 61.90%
```

Among the 619 changed responses:

```text
PPO wins:        303
SFT wins:        304
Reward ties:      12

PPO non-tie win rate: 49.92%
```

Therefore the result is not simply explained by PPO failing to change the policy.

Instead:

> PPO changed many outputs, but the changes did not have a consistent positive preference direction.

---

### Reward-Margin Analysis

Define

\[
m_{\mathrm{PPO}}
=
R(\mathrm{PPO})
-
R(\mathrm{SFT}).
\]

For PPO:

```text
Positive margins:       303
Negative margins:       304
Zero margins:           393

Mean margin:           -0.008694
Median margin:          0.000000
Mean absolute margin:   0.443561

Mean positive margin:  +0.717602
Mean negative margin:  -0.743841

Changed mean margin:   -0.014046
```

The positive and negative changes are nearly symmetric.

This suggests that the PPO policy updates were:

```text
stable,
relatively conservative,
but approximately directionless with respect to held-out reward preference.
```

There was no evidence of catastrophic PPO instability or reward collapse.

---

### DPO vs PPO

Using the same 1000-prompt held-out evaluation:

| Metric | DPO | PPO |
|---|---:|---:|
| Alignment-model wins | **404** | 303 |
| SFT wins | 322 | 304 |
| Ties | 274 | 393 |
| Non-tie win rate | **55.65%** | 49.92% |
| Mean reward improvement | **+0.093187** | -0.008694 |
| Identical responses | 260 | 381 |
| Changed responses | **740** | 619 |
| Changed mean margin | **+0.125928** | -0.014046 |
| Mean absolute margin | **0.623175** | 0.443561 |

DPO produced both:

1. more behavioral change, and
2. a clearer positive preference direction.

PPO changed fewer responses, and its positive and negative reward shifts were nearly balanced.

---

### Interpretation

The PPO run was technically successful:

- online rollouts worked correctly;
- the reward model provided online feedback;
- KL regularization constrained policy drift;
- the value model learned during training;
- PPO updates remained numerically stable;
- full-parameter PPO fit comfortably on a single RTX 3090.

However, stability alone did not translate into better held-out preference performance.

The final PPO policy changed 61.9% of held-out greedy responses, but those changes were almost evenly split between improvements and regressions according to the frozen reward model.

Under the current configuration:

\[
\boxed{
\text{SFT}\rightarrow\text{DPO}
\text{ produced a clearer preference improvement than }
\text{SFT}\rightarrow\text{PPO}.
}
\]

This result should not be interpreted as a general claim that DPO is superior to PPO. The two experiments used different optimization structures and training budgets, and PPO performance is sensitive to learning rate, KL strength, rollout budget, reward-model quality, and the number of online episodes.

For this project, the current PPO run is retained as a stable online-RLHF baseline rather than performing an extensive PPO hyperparameter sweep.

---

### Key Takeaways

- PPO differs fundamentally from DPO because responses are generated **online by the current policy**.
- The reward model provides the optimization target, while the value model performs return prediction and supports credit assignment.
- KL regularization controls drift from the original SFT policy.
- PPO clipping controls the size of each individual policy update.
- A critic initialized from a reward model is initially noisy but can learn substantially during PPO.
- Longer rollout budgets substantially improved EOS completion with negligible additional memory cost.
- Stable PPO optimization does not guarantee improved held-out preference performance.
- In this experiment, DPO produced a stronger and more consistent positive preference shift than PPO.

---

### Main Scripts

```text
scripts/prepare_ppo_dataset.py
    Prepare the prompt-only PPO dataset.

scripts/inspect_ppo_rollout.py
    Inspect online SFT rollouts and reward-model scores.

scripts/inspect_ppo_kl.py
    Verify policy/reference KL behavior.

scripts/inspect_ppo_advantage.py
    Inspect reward, value, GAE, return, and normalized advantage.

scripts/train_ppo_smoke.py
    Small PPO smoke tests and rollout-length ablations.

scripts/train_ppo.py
    Main PPO training script.

scripts/summarize_ppo_training.py
    Aggregate PPO training diagnostics.

scripts/evaluate_sft_vs_ppo.py
    Held-out SFT-vs-PPO evaluation.

scripts/compare_dpo_ppo_margins.py
    Compare DPO and PPO held-out reward-margin distributions.
```

## Day 5 — GRPO: Group-Relative Online Reinforcement Learning

### Goal

Day 5 implements **Group Relative Policy Optimization (GRPO)** as a second online reinforcement-learning baseline.

GRPO shares the online nature of PPO: responses are sampled from the current policy during training rather than taken from a fixed preference dataset.

The key difference is how the learning signal is constructed.

PPO uses a learned value model:

\[
R
\rightarrow
V(s)
\rightarrow
\mathrm{GAE}
\rightarrow
A_t
\]

whereas GRPO samples multiple completions for the same prompt and constructs relative advantages directly from their rewards:

\[
R_1,\ldots,R_G
\rightarrow
\text{group normalization}
\rightarrow
A_1,\ldots,A_G.
\]

This removes the need for a separate critic/value model.

For a group of \(G\) sampled completions with rewards \(r_i\),

\[
A_i
=
\frac{
r_i-\operatorname{mean}(r)
}{
\operatorname{std}(r)+\epsilon
}.
\]

A completion can therefore receive a positive advantage even when its absolute reward is negative, as long as it performs better than the other completions generated for the same prompt.

---

### GRPO Pipeline

```text
Prompt
  |
  v
Current SFT-initialized policy
  |
  +-- completion 1 -> reward 1
  +-- completion 2 -> reward 2
  +-- completion 3 -> reward 3
  +-- completion 4 -> reward 4
                         |
                         v
                 Group mean / std
                         |
                         v
                Relative advantages
                         |
                         v
                 GRPO policy update
```

The Day 2 reward model is reused as the frozen reward function.

Unlike PPO, this GRPO baseline does not use:

```text
Value model
GAE
Value loss
Reference-policy KL
```

The final configuration uses:

```text
beta = 0
```

so no frozen reference model is loaded.

---

### Prompt Dataset

The GRPO dataset is derived directly from the same prompt pool used for PPO.

The Day 4 tokenized prompts were decoded back into GRPOTrainer-compatible prompt strings without changing the underlying chat-template structure.

```text
Train prompts: 49,797
Eval prompts:     100
Columns:        ['prompt']
```

The prompt format retains the Qwen3 assistant generation prefix:

```text
<|im_start|>assistant
<think>

</think>
```

This keeps the SFT, PPO, and GRPO input formats aligned.

---

### Manual Group-Relative Advantage Inspection

Before training, one prompt was sampled four times using the SFT policy.

The resulting rewards were:

```text
Response 1: -1.679688
Response 2: -0.578125
Response 3: -1.773438
Response 4: -1.679688
```

with:

\[
\text{group mean}=-1.427734
\]

and:

\[
\text{group std}=0.568128.
\]

The corresponding normalized advantages were:

| Response | Reward | Advantage | Update direction |
|---|---:|---:|---|
| 1 | -1.6797 | -0.4434 | suppress |
| 2 | -0.5781 | **+1.4952** | reinforce |
| 3 | -1.7734 | -0.6084 | suppress |
| 4 | -1.6797 | -0.4434 | suppress |

Although every absolute reward was negative, Response 2 received a strong positive advantage because it was the best completion within the group.

This illustrates the central GRPO idea:

> optimization depends on relative performance among completions generated for the same prompt rather than on the absolute reward sign.

---

### Reward-Model Interface Test

A one-step dry run was performed with:

```text
learning_rate = 0
num_generations = 4
max_steps = 1
```

The full GRPOTrainer pipeline successfully executed:

```text
generation
-> reward-model scoring
-> group-relative normalization
-> GRPO loss
-> backward pass
```

The test produced:

```text
Mean reward:             -0.9170
Reward std:               0.4507
Zero-reward-std fraction: 0%
Gradient norm:            8.625
```

The policy parameters were unchanged, confirming that the zero-learning-rate dry run did not modify the SFT checkpoint.

---

### GRPO Smoke Training

A 16-step GRPO smoke test was then run with a nonzero learning rate.

```text
Learning rate:            1e-6
Group size:               4
Max completion length:    256
Loss type:                GRPO
Reward scaling:           group
KL beta:                  0
```

The policy weights changed successfully:

```text
Probe parameter:
model.layers.0.self_attn.q_proj.weight

Changed elements: 242 / 8192
Max absolute change: 6.68e-06
Policy updated: True
```

Smoke-test statistics:

| Metric | Value |
|---|---:|
| Mean reward | -0.1809 |
| Mean reward std | 1.3812 |
| Zero-std groups | 0% |
| Mean completion length | 184.69 |
| Truncated completion ratio | 39.06% |
| Mean gradient norm | 16.13 |
| Mean entropy | 0.4325 |

The nonzero reward variance confirmed that GRPO received meaningful relative learning signals across all sampled groups.

---

### Completion-Length Ablation

A controlled test compared maximum completion lengths of 256 and 384 tokens.

| Metric | 256 | 384 |
|---|---:|---:|
| Mean reward | -0.1809 | -0.0771 |
| Mean reward std | 1.3812 | 1.5699 |
| Zero-std groups | 0% | 0% |
| Mean completion length | 184.69 | 214.83 |
| Truncated completion ratio | **39.06%** | **34.38%** |
| Mean gradient norm | 16.13 | 17.29 |
| Max gradient norm | 104.5 | 97.0 |
| Mean entropy | 0.4325 | 0.3667 |

Increasing the maximum generation budget by 50% reduced truncation by only 4.68 percentage points.

Therefore:

```text
max_completion_length = 256
```

was retained for the main experiment.

---

### Main GRPO Configuration

```text
Policy initialization       Day 1 SFT
Reward model                frozen Day 2 RM

Train prompts               49,797
Training steps              1,024

Group size                  4
Sampled completions         ~4,096

Learning rate               1e-6
Scheduler                   linear

Max completion length       256
Temperature                 0.7
Top-p                       0.95

Loss type                   grpo
Reward scaling              group
GRPO iterations             1
Clip epsilon                0.2

Reference KL beta           0.0

Gradient clipping           1.0
Precision                   BF16
Training                    full-parameter
```

The approximately 4,096 sampled completions were chosen to provide a rollout budget comparable in scale to the 4,096 online responses used in the PPO baseline.

The final model is stored at:

```text
outputs/qwen3-0.6b-grpo
```

---

### Main GRPO Training Results

The full training run completed successfully:

```text
Steps:      1,024
Runtime:    ~69 minutes
```

Overall training statistics:

| Metric | Value |
|---|---:|
| Mean reward | +0.0528 |
| Mean reward std | 1.5193 |
| Zero-std groups | 0.78% |
| Mean completion length | 142.85 |
| Truncated completion ratio | 32.37% |
| Mean terminated length | 90.61 |
| Mean entropy | 0.4822 |
| Mean gradient norm | 29.21 |
| Max raw gradient norm | 488 |
| Mean step time | 3.98 s |

Reward diversity remained strong throughout training, with fewer than 1% of sampled groups having zero reward variance.

This indicates that meaningful group-relative advantages remained available for almost all training groups.

---

### Training Dynamics

| Metric | Q1 | Q2 | Q3 | Q4 |
|---|---:|---:|---:|---:|
| Mean reward | +0.0361 | +0.0725 | +0.0779 | +0.0246 |
| Reward std | 1.5903 | 1.4717 | 1.5153 | 1.4998 |
| Zero-std groups | 1.17% | 0.78% | 0.39% | 0.78% |
| Mean completion length | 141.25 | 144.12 | 150.61 | 135.42 |
| Truncated ratio | 31.54% | 31.15% | 35.45% | 31.35% |
| Entropy | 0.5018 | 0.4741 | 0.4700 | 0.4830 |

There was no evidence of reward-variance collapse or policy-entropy collapse.

Raw reward did not show a monotonic trend across training quarters. As with PPO, different quarters contain different shuffled prompts, so online training reward is not treated as a controlled model-quality metric.

The final model is instead evaluated on a fixed held-out benchmark.

---

### Held-Out Evaluation: SFT vs GRPO

The final GRPO checkpoint was evaluated on the same 1,000 held-out prompts used for DPO and PPO.

For each prompt:

```text
SFT  -> greedy response
GRPO -> greedy response

Frozen Day 2 RM -> score both responses
```

Results:

| Metric | Result |
|---|---:|
| GRPO wins | 301 |
| SFT wins | 302 |
| Ties | 397 |
| GRPO non-tie win rate | 49.92% |
| Mean SFT reward | 0.436594 |
| Mean GRPO reward | **0.458999** |
| Mean reward improvement | **+0.022405** |

GRPO therefore did not win more frequently than the SFT baseline, but produced a small positive shift in average reward.

---

### GRPO Response-Change Analysis

```text
Identical responses: 383 / 1000 = 38.30%
Changed responses:   617 / 1000 = 61.70%
```

Among changed responses:

```text
GRPO wins:       301
SFT wins:        302
Reward ties:      14

GRPO non-tie win rate: 49.92%
```

Therefore GRPO substantially changed model behavior, but the direction of these changes was nearly balanced in terms of win/loss frequency.

---

### GRPO Reward-Margin Analysis

Define:

\[
m_{\mathrm{GRPO}}
=
R(\mathrm{GRPO})
-
R(\mathrm{SFT}).
\]

The held-out reward margins were:

```text
Positive margins:        301
Negative margins:        302
Zero margins:            397

Mean margin:            +0.022405
Median margin:           0.000000

Mean absolute margin:    0.434851
Mean positive margin:   +0.759561
Mean negative margin:   -0.682859
```

Although the number of positive and negative comparisons was nearly identical, positive GRPO changes had a larger average magnitude than negative changes:

\[
0.7596 > 0.6829.
\]

This explains why the win rate remained approximately 50% while the average reward difference was slightly positive.

---

# Final Comparison — DPO vs PPO vs GRPO

All three post-training methods were evaluated using the same protocol:

```text
Same Day 1 SFT initialization
Same held-out test_prefs split
Same 1,000 evaluation prompts
Same greedy decoding
Same frozen Day 2 Reward Model
Same SFT baseline
```

The SFT mean reward was identical across all three evaluations:

\[
R_{\mathrm{SFT}}=0.436594.
\]

This provides a consistent reference point for the comparison.

## Held-Out Preference Results

| Metric | DPO | PPO | GRPO |
|---|---:|---:|---:|
| Aligned-model wins | **404** | 303 | 301 |
| SFT wins | 322 | 304 | 302 |
| Ties | 274 | 393 | 397 |
| Non-tie win rate | **55.65%** | 49.92% | 49.92% |
| Mean aligned reward | **0.529781** | 0.427900 | 0.458999 |
| Mean reward improvement | **+0.093187** | -0.008694 | **+0.022405** |
| Identical responses | 260 | 381 | 383 |
| Changed responses | **740** | 619 | 617 |

---

## Reward-Margin Comparison

| Metric | DPO | PPO | GRPO |
|---|---:|---:|---:|
| Positive margins | **404** | 303 | 301 |
| Negative margins | 322 | 304 | 302 |
| Zero margins | 274 | 393 | 397 |
| Mean margin | **+0.093187** | -0.008694 | +0.022405 |
| Mean positive margin | **+0.886586** | +0.717602 | +0.759561 |
| Mean negative margin | -0.822964 | -0.743841 | **-0.682859** |
| Mean absolute margin | **0.623175** | 0.443561 | 0.434851 |
| Changed-response mean margin | **+0.125928** | -0.014046 | — |

---

## Interpretation

### DPO

DPO produced the clearest positive preference shift.

It changed the largest number of held-out responses:

\[
740/1000=74\%.
\]

Among non-tied comparisons, DPO won:

\[
55.65\%.
\]

Its average reward improvement was:

\[
\boxed{+0.0932}.
\]

DPO therefore changed both the magnitude and the direction of model behavior more consistently toward the preference distribution represented by the reward model.

---

### PPO

PPO training was numerically stable and the critic learned successfully, but this did not translate into better held-out preference performance.

PPO changed:

\[
61.9\%
\]

of greedy responses, but among changed non-tied responses:

```text
PPO wins: 303
SFT wins: 304
```

The positive and negative changes were nearly symmetric, and negative changes were slightly larger in magnitude.

The final reward difference was:

\[
\boxed{-0.0087}.
\]

Thus, under this configuration, PPO produced behavioral change without a consistent positive preference direction.

---

### GRPO

GRPO also changed a substantial fraction of responses:

\[
61.7\%.
\]

Its win/loss frequency was almost exactly balanced:

```text
GRPO wins: 301
SFT wins:  302
```

However, GRPO's positive changes were larger on average than its negative changes:

\[
+0.7596
\quad\text{vs.}\quad
-0.6829.
\]

This produced a small positive mean reward shift:

\[
\boxed{+0.0224}.
\]

GRPO therefore performed better than the PPO baseline on average reward, although it did not demonstrate the consistent directional improvement observed with DPO.

---

## Overall Result

Under this specific experimental setup:

\[
\boxed{
\text{DPO} > \text{GRPO} > \text{PPO}
}
\]

in terms of mean held-out reward improvement:

\[
\boxed{
+0.0932
>
+0.0224
>
-0.0087
}
\]

However, this ordering should not be interpreted as a general ranking of the algorithms.

The three methods differ fundamentally in their training objectives and data usage:

```text
DPO
  -> offline chosen/rejected preference pairs

PPO
  -> online generation
  -> reward model
  -> value model / critic
  -> GAE

GRPO
  -> online grouped generation
  -> reward model
  -> group-relative advantages
```

The training budgets are also not perfectly equivalent. The PPO and GRPO experiments were designed as practical single-GPU baselines rather than exhaustive hyperparameter searches.

Therefore, the main conclusion is:

> In this Qwen3-0.6B + UltraFeedback experiment, DPO produced the strongest and most consistent held-out preference improvement. GRPO produced a smaller positive average reward shift through larger positive than negative reward margins, while the tested PPO configuration remained approximately neutral relative to the SFT baseline.

---

## Practical Takeaways

- **DPO** was the simplest method operationally and produced the strongest held-out preference improvement in this experiment.
- **PPO** required the most complex training architecture because it used a policy, reference policy, reward model, and trainable value model.
- **GRPO** removed the critic and replaced value-based credit assignment with relative comparison among multiple completions from the same prompt.
- GRPO maintained strong group-level reward diversity throughout training.
- Stable optimization alone did not guarantee improved held-out preference performance.
- Online RL methods were more sensitive to reward quality, rollout behavior, and training-budget choices than DPO.
- Using the same held-out prompts and frozen reward model was essential for making the three approaches meaningfully comparable.

---

## Day 5 Main Scripts

```text
scripts/inspect_grpo_group.py
    Manually inspect group rewards and relative advantages.

scripts/test_grpo_reward_interface.py
    Validate GRPOTrainer integration with the Day 2 reward model.

scripts/prepare_grpo_dataset.py
    Convert the prompt-only PPO dataset into GRPO format.

scripts/train_grpo_smoke.py
    Run GRPO smoke tests and completion-length ablations.

scripts/summarize_grpo_length_test.py
    Compare GRPO rollout-length configurations.

scripts/train_grpo.py
    Run the main GRPO experiment.

scripts/summarize_grpo_training.py
    Aggregate GRPO training diagnostics.

scripts/evaluate_sft_vs_grpo.py
    Evaluate SFT versus GRPO on 1,000 held-out prompts.

scripts/analyze_grpo_evaluation.py
    Analyze response changes and GRPO reward margins.
```

---

# Project-Level Summary

The complete project now covers the major stages of modern LLM post-training:

```text
Base Qwen3-0.6B
      |
      v
     SFT
      |
      +--------------------+
      |                    |
      v                    v
Reward Modeling           DPO
      |
      +--------------------+
      |                    |
      v                    v
     PPO                  GRPO
```

The implementation progressed from supervised learning and preference modeling to both offline and online alignment algorithms, with controlled held-out evaluation used to compare their behavioral effects.

The most important empirical observation was that **training stability and policy change are not sufficient evidence of successful alignment**. The direction and magnitude of held-out preference shifts must be measured explicitly.