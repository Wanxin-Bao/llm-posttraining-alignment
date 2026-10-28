import json
from pathlib import Path

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer


DATASET_NAME = "HuggingFaceH4/ultrafeedback_binarized"
TOKENIZER_PATH = "compat/qwen3-tokenizer-v4"

OUTPUT_DIR = "data/ppo_prompts_512"
STATS_PATH = "results/processed/day4_ppo_dataset_stats.json"

MAX_PROMPT_LENGTH = 512
EVAL_SAMPLES = 100
SEED = 42


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

    print("=" * 80)
    print("LOAD DATA")
    print("=" * 80)

    dataset = load_dataset(
        DATASET_NAME,
        split="train_prefs",
    )

    raw_count = len(dataset)

    print("Raw train pairs:", raw_count)

    # ---------------------------------------------------------
    # 1. Same positive-margin pool as Day 2 / Day 3
    # ---------------------------------------------------------

    dataset = dataset.filter(
        lambda x:
        x["score_chosen"] > x["score_rejected"]
    )

    positive_margin_count = len(dataset)

    print(
        "Positive-margin pairs:",
        positive_margin_count,
    )

    # ---------------------------------------------------------
    # 2. Remove duplicate prompts
    #
    # prompt_id and prompt text had the same unique count
    # in our audit, so prompt_id is sufficient.
    # ---------------------------------------------------------

    seen = set()
    keep_indices = []

    for i, prompt_id in enumerate(
        dataset["prompt_id"]
    ):
        if prompt_id not in seen:
            seen.add(prompt_id)
            keep_indices.append(i)

    dataset = dataset.select(
        keep_indices
    )

    deduplicated_count = len(dataset)

    print(
        "After prompt deduplication:",
        deduplicated_count,
    )

    # ---------------------------------------------------------
    # 3. Render + tokenize prompts exactly as PPO sees them
    # ---------------------------------------------------------

    def tokenize_prompt(example):

        rendered = render_prompt(
            tokenizer,
            example["prompt"],
        )

        input_ids = tokenizer(
            rendered,
            add_special_tokens=False,
        )["input_ids"]

        return {
            "input_ids": input_ids,
            "prompt_length": len(input_ids),
        }

    dataset = dataset.map(
        tokenize_prompt,
        num_proc=4,
    )

    # ---------------------------------------------------------
    # 4. Filter, DO NOT truncate
    #
    # A truncated instruction can silently change the task.
    # ---------------------------------------------------------

    dataset = dataset.filter(
        lambda x:
        x["prompt_length"]
        <= MAX_PROMPT_LENGTH
    )

    length_filtered_count = len(dataset)

    print(
        f"After <= {MAX_PROMPT_LENGTH} token filter:",
        length_filtered_count,
    )

    # ---------------------------------------------------------
    # 5. Shuffle deterministically
    # ---------------------------------------------------------

    dataset = dataset.shuffle(
        seed=SEED
    )

    # ---------------------------------------------------------
    # 6. Hold out a small internal PPO eval set
    #
    # This comes ONLY from train_prefs.
    # test_prefs remains untouched for final SFT/PPO evaluation.
    # ---------------------------------------------------------

    eval_dataset = dataset.select(
        range(EVAL_SAMPLES)
    )

    train_dataset = dataset.select(
        range(
            EVAL_SAMPLES,
            len(dataset),
        )
    )

    # PPOTrainer only needs input_ids.
    train_dataset = train_dataset.select_columns(
        ["input_ids"]
    )

    eval_dataset = eval_dataset.select_columns(
        ["input_ids"]
    )

    final_dataset = DatasetDict(
        {
            "train": train_dataset,
            "eval": eval_dataset,
        }
    )

    # ---------------------------------------------------------
    # 7. Save
    # ---------------------------------------------------------

    output_path = Path(
        OUTPUT_DIR
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_dataset.save_to_disk(
        OUTPUT_DIR
    )

    stats = {
        "source_dataset": DATASET_NAME,
        "source_split": "train_prefs",
        "raw_train_pairs": raw_count,
        "positive_margin_pairs": positive_margin_count,
        "deduplicated_prompts": deduplicated_count,
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "prompts_after_length_filter": length_filtered_count,
        "internal_eval_prompts": len(eval_dataset),
        "ppo_train_prompts": len(train_dataset),
        "seed": SEED,
        "final_external_eval_split": "test_prefs",
    }

    stats_path = Path(
        STATS_PATH
    )

    stats_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with stats_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            stats,
            f,
            indent=2,
        )

    # ---------------------------------------------------------
    # 8. Verification
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL PPO DATASET")
    print("=" * 80)

    print(
        "Train prompts:",
        len(train_dataset),
    )

    print(
        "Internal eval prompts:",
        len(eval_dataset),
    )

    train_lengths = [
        len(x)
        for x in train_dataset["input_ids"]
    ]

    eval_lengths = [
        len(x)
        for x in eval_dataset["input_ids"]
    ]

    print(
        "Train max length:",
        max(train_lengths),
    )

    print(
        "Eval max length:",
        max(eval_lengths),
    )

    print(
        "\nTrain columns:",
        train_dataset.column_names,
    )

    print(
        "Eval columns:",
        eval_dataset.column_names,
    )

    print(
        "\nSaved dataset:",
        OUTPUT_DIR,
    )

    print(
        "Saved statistics:",
        STATS_PATH,
    )

    # Decode one example to verify that the tokenized prompt
    # has exactly the structure expected by the policy.
    example_ids = (
        train_dataset[0]["input_ids"]
    )

    print("\n" + "=" * 80)
    print("DECODED TRAIN EXAMPLE")
    print("=" * 80)

    print(
        tokenizer.decode(
            example_ids,
            skip_special_tokens=False,
        )
    )


if __name__ == "__main__":
    main()
