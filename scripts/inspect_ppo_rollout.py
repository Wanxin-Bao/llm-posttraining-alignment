import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)


TOKENIZER_PATH = "compat/qwen3-tokenizer-v4"
POLICY_PATH = "outputs/qwen3-0.6b-sft"
REWARD_MODEL_PATH = "outputs/qwen3-0.6b-reward"

DATASET_NAME = "HuggingFaceH4/ultrafeedback_binarized"

NUM_PROMPTS = 3
MAX_NEW_TOKENS = 128

DEVICE = "cuda"


def score_response(
    reward_model,
    tokenizer,
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

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(DEVICE)

    with torch.inference_mode():
        logits = reward_model(**inputs).logits

    return logits.squeeze().float().item()


def main():
    set_seed(42)

    # ---------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        padding_side="left",
    )

    print("Tokenizer:", type(tokenizer).__name__)
    print("PAD:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("EOS:", tokenizer.eos_token, tokenizer.eos_token_id)

    # ---------------------------------------------------------
    # Load policy
    # ---------------------------------------------------------

    print("\nLoading SFT policy...")

    policy = AutoModelForCausalLM.from_pretrained(
        POLICY_PATH,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(DEVICE)

    policy.eval()

    print("Policy loaded.")

    # ---------------------------------------------------------
    # Load reward model
    # ---------------------------------------------------------

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

    print("Reward model loaded.")

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset = load_dataset(
        DATASET_NAME,
        split="test_prefs",
    )

    # Same clean preference rule as Day 2 / Day 3.
    dataset = dataset.filter(
        lambda x:
        x["score_chosen"] > x["score_rejected"]
    )

    dataset = dataset.select(
        range(min(NUM_PROMPTS, len(dataset)))
    )

    print(
        "\nNumber of rollout prompts:",
        len(dataset),
    )

    # ---------------------------------------------------------
    # Online rollout
    # ---------------------------------------------------------

    for i, sample in enumerate(dataset):

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

        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
        ).to(DEVICE)

        prompt_length = inputs["input_ids"].shape[1]

        # PPO rollouts should be sampled rather than greedy.
        with torch.inference_mode():
            output = policy.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response_ids = output[
            0,
            prompt_length:
        ]

        response = tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
        ).strip()

        reward = score_response(
            reward_model,
            tokenizer,
            prompt,
            response,
        )

        generated_eos = (
            tokenizer.eos_token_id
            in response_ids.tolist()
        )

        print("\n" + "=" * 80)
        print(f"ROLLOUT {i + 1}")
        print("=" * 80)

        print("\nPROMPT:")
        print(prompt)

        print("\nPOLICY RESPONSE:")
        print(response)

        print("\nRM REWARD:")
        print(f"{reward:.6f}")

        print("\nROLLOUT INFO:")
        print(
            "Generated tokens:",
            len(response_ids),
        )
        print(
            "Ended with EOS:",
            generated_eos,
        )

        print(
            "\nDataset source scores:",
            f"chosen={sample['score_chosen']}, "
            f"rejected={sample['score_rejected']}",
        )


if __name__ == "__main__":
    main()
