from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer


TOKENIZER_PATH = "compat/qwen3-tokenizer-v4"
PPO_DATASET_PATH = "data/ppo_prompts_512"
OUTPUT_PATH = "data/grpo_prompts_512"


def convert_split(split, tokenizer):
    prompts = []

    for example in split:
        text = tokenizer.decode(
            example["input_ids"],
            skip_special_tokens=False,
        )

        prompts.append(text)

    return Dataset.from_dict(
        {
            "prompt": prompts,
        }
    )


def main():

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
    )

    dataset = load_from_disk(
        PPO_DATASET_PATH
    )

    grpo_dataset = DatasetDict(
        {
            "train": convert_split(
                dataset["train"],
                tokenizer,
            ),
            "eval": convert_split(
                dataset["eval"],
                tokenizer,
            ),
        }
    )

    print("=" * 80)
    print("GRPO DATASET")
    print("=" * 80)

    print(
        "Train prompts:",
        len(grpo_dataset["train"]),
    )

    print(
        "Eval prompts:",
        len(grpo_dataset["eval"]),
    )

    print(
        "Train columns:",
        grpo_dataset["train"].column_names,
    )

    print()
    print("Example:")
    print("-" * 80)
    print(
        grpo_dataset["train"][0]["prompt"]
    )
    print("-" * 80)

    grpo_dataset.save_to_disk(
        OUTPUT_PATH
    )

    print()
    print(
        "Saved to:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
