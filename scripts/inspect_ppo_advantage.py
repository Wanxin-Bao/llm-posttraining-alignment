import torch
import torch.nn.functional as F

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)

from trl.trainer.utils import get_reward


TOKENIZER_PATH = "compat/qwen3-tokenizer-v4"

POLICY_PATH = "outputs/qwen3-0.6b-sft"
REFERENCE_PATH = "outputs/qwen3-0.6b-sft"
REWARD_MODEL_PATH = "outputs/qwen3-0.6b-reward"
VALUE_MODEL_PATH = "outputs/qwen3-0.6b-reward"

DATASET_NAME = "HuggingFaceH4/ultrafeedback_binarized"

DEVICE = "cuda"

MAX_NEW_TOKENS = 128

KL_COEF = 0.05

# TRL PPO defaults
GAMMA = 1.0
LAM = 0.95


def get_response_logprobs(
    model,
    full_input_ids,
    prompt_length,
):
    with torch.inference_mode():
        outputs = model(
            input_ids=full_input_ids
        )

    logits = outputs.logits[:, :-1, :].float()
    labels = full_input_ids[:, 1:]

    log_probs = F.log_softmax(
        logits,
        dim=-1,
    )

    token_logprobs = torch.gather(
        log_probs,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)

    # First response token is predicted from
    # the last prompt-token position.
    response_logprobs = token_logprobs[
        :,
        prompt_length - 1:
    ]

    return response_logprobs.squeeze(0)


def compute_gae(
    rewards,
    values,
):
    """
    Same core GAE recursion used by TRL PPO.
    """

    advantages = torch.zeros_like(rewards)
    deltas = torch.zeros_like(rewards)

    lastgaelam = torch.tensor(
        0.0,
        device=rewards.device,
    )

    for t in reversed(range(len(rewards))):

        if t < len(rewards) - 1:
            next_value = values[t + 1]
        else:
            next_value = torch.tensor(
                0.0,
                device=rewards.device,
            )

        delta = (
            rewards[t]
            + GAMMA * next_value
            - values[t]
        )

        lastgaelam = (
            delta
            + GAMMA * LAM * lastgaelam
        )

        deltas[t] = delta
        advantages[t] = lastgaelam

    returns = advantages + values

    # TRL whitens advantages before PPO optimization.
    if len(advantages) > 1:
        advantage_mean = advantages.mean()
        advantage_var = advantages.var(
            unbiased=True
        )

        whitened_advantages = (
            advantages - advantage_mean
        ) / torch.sqrt(
            advantage_var + 1e-8
        )
    else:
        whitened_advantages = advantages.clone()

    return (
        deltas,
        advantages,
        returns,
        whitened_advantages,
    )


def main():
    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        padding_side="left",
    )

    # ---------------------------------------------------------
    # Models
    # ---------------------------------------------------------

    print("Loading policy...")

    policy = AutoModelForCausalLM.from_pretrained(
        POLICY_PATH,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(DEVICE)

    policy.eval()

    print("Loading reference...")

    reference = AutoModelForCausalLM.from_pretrained(
        REFERENCE_PATH,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(DEVICE)

    reference.eval()

    for p in reference.parameters():
        p.requires_grad_(False)

    print("Loading reward model...")

    reward_model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            REWARD_MODEL_PATH,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(DEVICE)
    )

    reward_model.eval()

    for p in reward_model.parameters():
        p.requires_grad_(False)

    print("Loading value model...")

    value_model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            VALUE_MODEL_PATH,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(DEVICE)
    )

    value_model.eval()

    # For this inspection only.
    # During PPO training the value model is trainable.
    for p in value_model.parameters():
        p.requires_grad_(False)

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset = load_dataset(
        DATASET_NAME,
        split="test_prefs",
    )

    dataset = dataset.filter(
        lambda x:
        x["score_chosen"]
        > x["score_rejected"]
    )

    sample = dataset[0]

    prompt = sample["prompt"]

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    prompt_inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
    ).to(DEVICE)

    prompt_length = (
        prompt_inputs["input_ids"].shape[1]
    )

    # ---------------------------------------------------------
    # 1. Policy rollout
    # ---------------------------------------------------------

    with torch.inference_mode():
        generated = policy.generate(
            **prompt_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response_ids = generated[
        0,
        prompt_length:
    ]

    response = tokenizer.decode(
        response_ids,
        skip_special_tokens=True,
    ).strip()

    full_input_ids = generated

    ended_with_eos = (
        tokenizer.eos_token_id
        in response_ids.tolist()
    )

    # ---------------------------------------------------------
    # 2. Policy / reference log-probs
    # ---------------------------------------------------------

    policy_logps = get_response_logprobs(
        policy,
        full_input_ids,
        prompt_length,
    )

    reference_logps = get_response_logprobs(
        reference,
        full_input_ids,
        prompt_length,
    )

    log_ratio = (
        policy_logps
        - reference_logps
    )

    kl_rewards = (
        -KL_COEF * log_ratio
    )

    # ---------------------------------------------------------
    # 3. Reward Model score
    #
    # Use TRL's own get_reward() implementation.
    # ---------------------------------------------------------

    with torch.inference_mode():
        _, rm_score_tensor, _ = get_reward(
            reward_model,
            full_input_ids,
            tokenizer.pad_token_id,
            prompt_length,
        )

    rm_score = (
        rm_score_tensor
        .squeeze()
        .float()
    )

    # ---------------------------------------------------------
    # 4. Value predictions
    #
    # get_reward() returns a score for every token.
    #
    # TRL aligns response values using:
    # context_length - 1 : -1
    # ---------------------------------------------------------

    with torch.inference_mode():
        full_values, _, _ = get_reward(
            value_model,
            full_input_ids,
            tokenizer.pad_token_id,
            prompt_length,
        )

    values = full_values[
        :,
        prompt_length - 1:-1,
    ].squeeze(0).squeeze(-1).float()

    assert len(values) == len(response_ids)

    # ---------------------------------------------------------
    # 5. Construct token rewards
    #
    # Initially:
    # KL reward ≈ 0 everywhere.
    #
    # RM score is added only to the last response token.
    # ---------------------------------------------------------

    rewards = kl_rewards.clone().float()

    rewards[-1] += rm_score

    # ---------------------------------------------------------
    # 6. GAE
    # ---------------------------------------------------------

    (
        deltas,
        advantages,
        returns,
        whitened_advantages,
    ) = compute_gae(
        rewards,
        values,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\n" + "=" * 100)
    print("PPO VALUE / ADVANTAGE INSPECTION")
    print("=" * 100)

    print("\nPROMPT:")
    print(prompt)

    print("\nRESPONSE:")
    print(response)

    print("\nSUMMARY")
    print("-" * 100)

    print(
        f"Response tokens:           "
        f"{len(response_ids)}"
    )

    print(
        f"Ended with EOS:            "
        f"{ended_with_eos}"
    )

    print(
        f"RM score:                  "
        f"{rm_score.item():+.6f}"
    )

    print(
        f"Sampled KL:                "
        f"{log_ratio.sum().item():+.8f}"
    )

    print(
        f"Total KL reward:           "
        f"{kl_rewards.sum().item():+.8f}"
    )

    print(
        f"Total token reward:        "
        f"{rewards.sum().item():+.6f}"
    )

    print(
        f"Mean initial value:        "
        f"{values.mean().item():+.6f}"
    )

    print(
        f"Mean raw advantage:        "
        f"{advantages.mean().item():+.6f}"
    )

    print(
        f"Std raw advantage:         "
        f"{advantages.std().item():+.6f}"
    )

    # ---------------------------------------------------------
    # Token table
    # ---------------------------------------------------------

    print("\nTOKEN-LEVEL CREDIT ASSIGNMENT")
    print("-" * 100)

    print(
        f"{'idx':>4} "
        f"{'token':<17} "
        f"{'reward':>10} "
        f"{'V(s)':>10} "
        f"{'delta':>10} "
        f"{'adv(raw)':>11} "
        f"{'return':>10} "
        f"{'adv(norm)':>11}"
    )

    print("-" * 100)

    # First 8 + last 8 tokens.
    indices = list(
        range(min(8, len(response_ids)))
    )

    tail_start = max(
        8,
        len(response_ids) - 8,
    )

    indices += list(
        range(tail_start, len(response_ids))
    )

    for j in indices:

        token_id = response_ids[j].item()

        token = repr(
            tokenizer.decode([token_id])
        )

        if len(token) > 16:
            token = token[:13] + "..."

        print(
            f"{j:>4} "
            f"{token:<17} "
            f"{rewards[j].item():>+10.4f} "
            f"{values[j].item():>+10.4f} "
            f"{deltas[j].item():>+10.4f} "
            f"{advantages[j].item():>+11.4f} "
            f"{returns[j].item():>+10.4f} "
            f"{whitened_advantages[j].item():>+11.4f}"
        )

    if len(response_ids) > 16:
        print("\n... middle tokens omitted ...")


if __name__ == "__main__":
    main()