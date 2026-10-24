from datasets import load_dataset
from transformers import AutoTokenizer


DATASET_NAME = "HuggingFaceH4/ultrafeedback_binarized"
MODEL_PATH = "outputs/qwen3-0.6b-sft"


def print_messages(messages):
    for message in messages:
        role = message["role"].upper()
        content = message["content"]

        print(f"\n[{role}]")
        print(content[:1200])


def main():
    print(f"Loading dataset: {DATASET_NAME}")

    dataset = load_dataset(DATASET_NAME)

    print("\n===== DATASET STRUCTURE =====")
    print(dataset)

    train = dataset["train_prefs"]
    test = dataset["test_prefs"]

    print("\nTrain preference pairs:", len(train))
    print("Test preference pairs:", len(test))
    print("Columns:", train.column_names)

    print("\n===== FIRST 3 PREFERENCE PAIRS =====")

    for i in range(3):
        sample = train[i]

        print("\n" + "=" * 100)
        print(f"PAIR {i}")

        print("\nPROMPT:")
        print(sample["prompt"])

        print("\n----- CHOSEN -----")
        print_messages(sample["chosen"])

        print("\nChosen score:", sample["score_chosen"])

        print("\n----- REJECTED -----")
        print_messages(sample["rejected"])

        print("\nRejected score:", sample["score_rejected"])

        print(
            "\nScore margin:",
            sample["score_chosen"] - sample["score_rejected"],
        )

    # ---------------------------------------------------------
    # Sequence length analysis
    # ---------------------------------------------------------

    print("\n===== TOKEN LENGTH ANALYSIS =====")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    chosen_lengths = []
    rejected_lengths = []

    # Analyze first 2000 training pairs.
    num_samples = min(2000, len(train))

    for i in range(num_samples):
        sample = train[i]

        chosen_text = tokenizer.apply_chat_template(
            sample["chosen"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        rejected_text = tokenizer.apply_chat_template(
            sample["rejected"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        chosen_ids = tokenizer(
            chosen_text,
            add_special_tokens=False,
        )["input_ids"]

        rejected_ids = tokenizer(
            rejected_text,
            add_special_tokens=False,
        )["input_ids"]

        chosen_lengths.append(len(chosen_ids))
        rejected_lengths.append(len(rejected_ids))

    all_lengths = chosen_lengths + rejected_lengths
    all_lengths.sort()

    def percentile(values, p):
        index = int((len(values) - 1) * p)
        return values[index]

    print("Preference pairs analyzed:", num_samples)

    print("\nCombined chosen/rejected lengths:")
    print("Min   :", min(all_lengths))
    print("Median:", percentile(all_lengths, 0.50))
    print("P90   :", percentile(all_lengths, 0.90))
    print("P95   :", percentile(all_lengths, 0.95))
    print("P99   :", percentile(all_lengths, 0.99))
    print("Max   :", max(all_lengths))

    # ---------------------------------------------------------
    # Preference score margins
    # ---------------------------------------------------------

    margins = []

    for i in range(num_samples):
        margin = (
            train[i]["score_chosen"]
            - train[i]["score_rejected"]
        )

        margins.append(margin)

    margins.sort()

    print("\n===== PREFERENCE SCORE MARGINS =====")

    print("Min margin   :", min(margins))
    print("Median margin:", percentile(margins, 0.50))
    print("P90 margin   :", percentile(margins, 0.90))
    print("Max margin   :", max(margins))


if __name__ == "__main__":
    main()
