from datasets import load_dataset
from transformers import AutoTokenizer


DATASET_NAME = "trl-lib/Capybara"
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"


def main():
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)

    print("\n===== DATASET STRUCTURE =====")
    print(dataset)

    train_dataset = dataset["train"]

    print("\nTrain samples:", len(train_dataset))
    print("Columns:", train_dataset.column_names)

    print("\n===== FIRST 3 RAW SAMPLES =====")

    for i in range(3):
        sample = train_dataset[i]

        print("\n" + "=" * 80)
        print(f"SAMPLE {i}")
        print("Source:", sample.get("source"))
        print("Num turns:", sample.get("num_turns"))

        for message in sample["messages"]:
            print(f"\n[{message['role'].upper()}]")
            print(message["content"][:1000])

    print("\n===== CHAT TEMPLATE EXAMPLE =====")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    sample = train_dataset[0]

    rendered = tokenizer.apply_chat_template(
        sample["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

    print(rendered)

    print("\n===== TOKEN LENGTHS =====")

    lengths = []

    for i in range(min(1000, len(train_dataset))):
        rendered = tokenizer.apply_chat_template(
            train_dataset[i]["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

        ids = tokenizer(
            rendered,
            add_special_tokens=False,
        )["input_ids"]

        lengths.append(len(ids))

    lengths.sort()

    def percentile(p):
        index = int((len(lengths) - 1) * p)
        return lengths[index]

    print("Samples analyzed:", len(lengths))
    print("Min tokens:", min(lengths))
    print("Median tokens:", percentile(0.50))
    print("P90 tokens:", percentile(0.90))
    print("P95 tokens:", percentile(0.95))
    print("Max tokens:", max(lengths))


if __name__ == "__main__":
    main()
