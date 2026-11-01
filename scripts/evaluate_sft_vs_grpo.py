import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


SFT_MODEL = "outputs/qwen3-0.6b-sft"
GRPO_MODEL = "outputs/qwen3-0.6b-grpo"
REWARD_MODEL = "outputs/qwen3-0.6b-reward"

DATASET_NAME = "HuggingFaceH4/ultrafeedback_binarized"

NUM_PROMPTS = 1000
MAX_NEW_TOKENS = 256


def generate(model, tokenizer, prompt):
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[
        0,
        inputs["input_ids"].shape[1]:,
    ]

    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()


def reward_score(
    reward_model,
    reward_tokenizer,
    prompt,
    response,
):
    messages = [
        {
            "role": "user",
            "content": prompt,
        },
        {
            "role": "assistant",
            "content": response,
        },
    ]

    text = reward_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    inputs = reward_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(reward_model.device)

    with torch.inference_mode():
        output = reward_model(**inputs)

    return output.logits.squeeze().item()


def main():
    print("Loading held-out prompts...")

    dataset = load_dataset(
        DATASET_NAME,
        split="test_prefs",
    )

    dataset = dataset.filter(
        lambda x: x["score_chosen"]
        > x["score_rejected"]
    )

    dataset = dataset.select(
        range(min(NUM_PROMPTS, len(dataset)))
    )

    # -----------------------------------------------------
    # Tokenizers
    # -----------------------------------------------------

    sft_tokenizer = AutoTokenizer.from_pretrained(
        SFT_MODEL
    )

    grpo_tokenizer = AutoTokenizer.from_pretrained(
        GRPO_MODEL
    )

    reward_tokenizer = AutoTokenizer.from_pretrained(
        REWARD_MODEL
    )

    # -----------------------------------------------------
    # Models
    # -----------------------------------------------------

    print("Loading SFT model...")

    sft_model = AutoModelForCausalLM.from_pretrained(
        SFT_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    sft_model.eval()

    print("Loading GRPO model...")

    grpo_model = AutoModelForCausalLM.from_pretrained(
        GRPO_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    grpo_model.eval()

    print("Loading reward model...")

    reward_model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            REWARD_MODEL,
            dtype=torch.bfloat16,
            device_map="auto",
        )
    )
    reward_model.eval()

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    results = []

    grpo_wins = 0
    sft_wins = 0
    ties = 0

    sum_sft_reward = 0.0
    sum_grpo_reward = 0.0

    for i, sample in enumerate(dataset):
        prompt = sample["prompt"]

        sft_response = generate(
            sft_model,
            sft_tokenizer,
            prompt,
        )

        grpo_response = generate(
            grpo_model,
            grpo_tokenizer,
            prompt,
        )

        sft_reward = reward_score(
            reward_model,
            reward_tokenizer,
            prompt,
            sft_response,
        )

        grpo_reward = reward_score(
            reward_model,
            reward_tokenizer,
            prompt,
            grpo_response,
        )

        margin = grpo_reward - sft_reward

        if margin > 0:
            winner = "GRPO"
            grpo_wins += 1

        elif margin < 0:
            winner = "SFT"
            sft_wins += 1

        else:
            winner = "TIE"
            ties += 1

        sum_sft_reward += sft_reward
        sum_grpo_reward += grpo_reward

        results.append(
            {
                "id": i,
                "prompt": prompt,
                "sft_response": sft_response,
                "grpo_response": grpo_response,
                "sft_reward": sft_reward,
                "grpo_reward": grpo_reward,
                "reward_margin": margin,
                "winner": winner,
            }
        )

        print(
            f"[{i + 1:03d}/{len(dataset)}] "
            f"SFT={sft_reward:.3f} "
            f"GRPO={grpo_reward:.3f} "
            f"margin={margin:+.3f} "
            f"winner={winner}"
        )

    n = len(results)

    print("\n" + "=" * 80)
    print("SFT vs GRPO EXTERNAL REWARD EVALUATION")
    print("=" * 80)

    print("Prompts:", n)

    print(
        f"GRPO wins: {grpo_wins} "
        f"({100 * grpo_wins / n:.2f}%)"
    )

    print(
        f"SFT wins: {sft_wins} "
        f"({100 * sft_wins / n:.2f}%)"
    )

    print(
        f"Ties: {ties} "
        f"({100 * ties / n:.2f}%)"
    )

    print(
        "Mean SFT reward:",
        sum_sft_reward / n,
    )

    print(
        "Mean GRPO reward:",
        sum_grpo_reward / n,
    )

    print(
        "Mean reward improvement:",
        (
            sum_grpo_reward
            - sum_sft_reward
        ) / n,
    )

    output_path = Path(
        "results/processed/"
        "sft_vs_grpo_reward_evaluation.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
