import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer


DATASET_NAME = "HuggingFaceH4/ultrafeedback_binarized"
TOKENIZER_PATH = "compat/qwen3-tokenizer-v4"

SAMPLE_PRINT = 3


def render_prompt(tokenizer, prompt):
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def main():

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        padding_side="left",
    )

    dataset = load_dataset(
        DATASET_NAME,
        split="train_prefs",
    )

    print("=" * 80)
    print("RAW DATA")
    print("=" * 80)

    print("Raw train pairs:", len(dataset))

    # Same clean preference pool as Day 2 / Day 3.
    dataset = dataset.filter(
        lambda x:
        x["score_chosen"]
        > x["score_rejected"]
    )

    print(
        "Positive-margin pairs:",
        len(dataset),
    )

    # ---------------------------------------------------------
    # Duplicate audit
    # ---------------------------------------------------------

    prompts = dataset["prompt"]

    unique_prompts = set(prompts)

    print("\n" + "=" * 80)
    print("DUPLICATE AUDIT")
    print("=" * 80)

    print("Total prompts:", len(prompts))
    print("Unique prompts:", len(unique_prompts))
    print(
        "Duplicate rows:",
        len(prompts) - len(unique_prompts),
    )

    if "prompt_id" in dataset.column_names:
        prompt_ids = dataset["prompt_id"]

        print(
            "Unique prompt IDs:",
            len(set(prompt_ids)),
        )

    # ---------------------------------------------------------
    # Render + tokenize exactly as PPO will see them
    # ---------------------------------------------------------

    lengths = []

    for prompt in prompts:

        rendered = render_prompt(
            tokenizer,
            prompt,
        )

        ids = tokenizer(
            rendered,
            add_special_tokens=False,
        )["input_ids"]

        lengths.append(len(ids))

    lengths = np.asarray(lengths)

    print("\n" + "=" * 80)
    print("RENDERED PROMPT TOKEN LENGTH")
    print("=" * 80)

    print("Count :", len(lengths))
    print("Min   :", int(lengths.min()))
    print("Median:", int(np.percentile(lengths, 50)))
    print("P90   :", int(np.percentile(lengths, 90)))
    print("P95   :", int(np.percentile(lengths, 95)))
    print("P99   :", int(np.percentile(lengths, 99)))
    print("Max   :", int(lengths.max()))

    for threshold in [
        128,
        256,
        384,
        512,
        768,
        1024,
    ]:
        count = int(
            np.sum(lengths > threshold)
        )

        pct = (
            100.0 * count / len(lengths)
        )

        print(
            f"> {threshold:4d}: "
            f"{count:6d} "
            f"({pct:6.2f}%)"
        )

    # ---------------------------------------------------------
    # Show rendered examples
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("EXAMPLES")
    print("=" * 80)

    for i in range(
        min(SAMPLE_PRINT, len(dataset))
    ):

        prompt = dataset[i]["prompt"]

        rendered = render_prompt(
            tokenizer,
            prompt,
        )

        ids = tokenizer(
            rendered,
            add_special_tokens=False,
        )["input_ids"]

        print(
            f"\n--- Example {i + 1} "
            f"({len(ids)} tokens) ---"
        )

        print(rendered)


if __name__ == "__main__":
    main()
