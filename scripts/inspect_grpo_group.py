import torch

from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


TOKENIZER_PATH = "compat/qwen3-tokenizer-v4"

POLICY_PATH = "outputs/qwen3-0.6b-sft"
REWARD_MODEL_PATH = "outputs/qwen3-0.6b-reward"

DATASET_PATH = "data/ppo_prompts_512"

GROUP_SIZE = 4
MAX_NEW_TOKENS = 256

TEMPERATURE = 0.7
TOP_P = 0.95

SEED = 42

DEVICE = "cuda"


def score_response(
    reward_model,
    prompt_ids,
    completion_ids,
):
    """
    Score one complete prompt + response with the
    Day-2 reward model.
    """

    full_ids = torch.cat(
        [
            prompt_ids,
            completion_ids,
        ],
        dim=1,
    )

    attention_mask = torch.ones_like(
        full_ids
    )

    with torch.no_grad():
        outputs = reward_model(
            input_ids=full_ids,
            attention_mask=attention_mask,
        )

    reward = (
        outputs.logits
        .squeeze()
        .float()
        .item()
    )

    return reward


def main():

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # ---------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        padding_side="left",
    )

    print("=" * 80)
    print("TOKENIZER")
    print("=" * 80)

    print(
        "PAD:",
        tokenizer.pad_token,
        tokenizer.pad_token_id,
    )

    print(
        "EOS:",
        tokenizer.eos_token,
        tokenizer.eos_token_id,
    )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset = load_from_disk(
        DATASET_PATH
    )

    # Use one held-out internal PPO prompt.
    example = dataset["eval"][0]

    prompt_ids = torch.tensor(
        example["input_ids"],
        dtype=torch.long,
    ).unsqueeze(0)

    prompt_length = prompt_ids.shape[1]

    print()
    print("=" * 80)
    print("PROMPT")
    print("=" * 80)

    print(
        tokenizer.decode(
            prompt_ids[0],
            skip_special_tokens=False,
        )
    )

    print()
    print(
        "Prompt tokens:",
        prompt_length,
    )

    # ---------------------------------------------------------
    # Models
    # ---------------------------------------------------------

    model_kwargs = {
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
    }

    print()
    print("Loading SFT policy...")

    policy = (
        AutoModelForCausalLM
        .from_pretrained(
            POLICY_PATH,
            **model_kwargs,
        )
        .to(DEVICE)
    )

    policy.eval()

    print("Loading reward model...")

    reward_model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            REWARD_MODEL_PATH,
            num_labels=1,
            **model_kwargs,
        )
        .to(DEVICE)
    )

    reward_model.eval()

    prompt_ids = prompt_ids.to(DEVICE)

    # ---------------------------------------------------------
    # Generate a GRPO group
    # ---------------------------------------------------------

    rewards = []
    responses = []
    completion_lengths = []
    eos_flags = []

    print()
    print("=" * 80)
    print(
        f"GENERATING {GROUP_SIZE} RESPONSES"
    )
    print("=" * 80)

    for i in range(GROUP_SIZE):

        with torch.no_grad():

            output_ids = policy.generate(
                input_ids=prompt_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        completion_ids = output_ids[
            :,
            prompt_length:
        ]

        completion_length = (
            completion_ids.shape[1]
        )

        ended_with_eos = bool(
            completion_ids[0, -1].item()
            == tokenizer.eos_token_id
        )

        response = tokenizer.decode(
            completion_ids[0],
            skip_special_tokens=True,
        )

        reward = score_response(
            reward_model,
            prompt_ids,
            completion_ids,
        )

        rewards.append(reward)
        responses.append(response)
        completion_lengths.append(
            completion_length
        )
        eos_flags.append(
            ended_with_eos
        )

        print()
        print("-" * 80)
        print(f"RESPONSE {i + 1}")
        print("-" * 80)

        print(response)

        print()
        print(
            f"Tokens: {completion_length}"
        )

        print(
            f"EOS: {ended_with_eos}"
        )

        print(
            f"Reward: {reward:+.6f}"
        )

    # ---------------------------------------------------------
    # GRPO group-relative advantage
    # ---------------------------------------------------------

    rewards_tensor = torch.tensor(
        rewards,
        dtype=torch.float32,
    )

    group_mean = (
        rewards_tensor.mean()
    )

    # torch.std() default uses sample std (Bessel correction).
    group_std = (
        rewards_tensor.std()
    )

    if group_std.item() > 0:

        advantages = (
            rewards_tensor
            - group_mean
        ) / (
            group_std
            + 1e-4
        )

    else:

        advantages = torch.zeros_like(
            rewards_tensor
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print()
    print("=" * 80)
    print("GRPO GROUP SUMMARY")
    print("=" * 80)

    print(
        f"Group mean reward: "
        f"{group_mean.item():+.6f}"
    )

    print(
        f"Group std reward:  "
        f"{group_std.item():.6f}"
    )

    print()

    print(
        f"{'ID':<5}"
        f"{'Reward':>12}"
        f"{'Advantage':>14}"
        f"{'Tokens':>10}"
        f"{'EOS':>8}"
    )

    print("-" * 49)

    for i in range(GROUP_SIZE):

        print(
            f"{i + 1:<5}"
            f"{rewards[i]:>+12.6f}"
            f"{advantages[i].item():>+14.6f}"
            f"{completion_lengths[i]:>10}"
            f"{str(eos_flags[i]):>8}"
        )

    print()
    print(
        "Mean advantage:",
        f"{advantages.mean().item():+.8f}",
    )

    best_idx = int(
        torch.argmax(
            rewards_tensor
        ).item()
    )

    worst_idx = int(
        torch.argmin(
            rewards_tensor
        ).item()
    )

    print()
    print(
        "Highest-reward response:",
        best_idx + 1,
    )

    print(
        "Lowest-reward response:",
        worst_idx + 1,
    )

    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    for i in range(GROUP_SIZE):

        advantage = (
            advantages[i].item()
        )

        if advantage > 0:

            direction = (
                "increase probability"
            )

        elif advantage < 0:

            direction = (
                "decrease probability"
            )

        else:

            direction = (
                "approximately no preference signal"
            )

        print(
            f"Response {i + 1}: "
            f"A={advantage:+.4f} "
            f"-> {direction}"
        )


if __name__ == "__main__":
    main()
