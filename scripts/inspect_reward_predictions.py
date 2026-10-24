from datasets import load_dataset
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_PATH = "outputs/qwen3-0.6b-reward"
DATASET_NAME = "HuggingFaceH4/ultrafeedback_binarized"

NUM_CORRECT = 3
NUM_WRONG = 3


def prepare_text(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )


def reward_score(model, tokenizer, messages):
    text = prepare_text(tokenizer, messages)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(model.device)

    with torch.inference_mode():
        outputs = model(**inputs)

    return outputs.logits.squeeze().item()


def short(text, max_chars=600):
    text = text.replace("\n", " ")
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("Loading reward model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    dataset = load_dataset(
        DATASET_NAME,
        split="test_prefs",
    )

    # Keep the same clean preference definition used in training.
    dataset = dataset.filter(
        lambda x: x["score_chosen"] > x["score_rejected"]
    )

    correct_examples = []
    wrong_examples = []

    for sample in dataset:
        chosen = sample["chosen"]
        rejected = sample["rejected"]

        r_chosen = reward_score(
            model,
            tokenizer,
            chosen,
        )

        r_rejected = reward_score(
            model,
            tokenizer,
            rejected,
        )

        margin = r_chosen - r_rejected
        correct = margin > 0

        result = {
            "prompt": sample["prompt"],
            "chosen": chosen[-1]["content"],
            "rejected": rejected[-1]["content"],
            "score_chosen": sample["score_chosen"],
            "score_rejected": sample["score_rejected"],
            "reward_chosen": r_chosen,
            "reward_rejected": r_rejected,
            "margin": margin,
        }

        if correct and len(correct_examples) < NUM_CORRECT:
            correct_examples.append(result)

        if not correct and len(wrong_examples) < NUM_WRONG:
            wrong_examples.append(result)

        if (
            len(correct_examples) >= NUM_CORRECT
            and len(wrong_examples) >= NUM_WRONG
        ):
            break

    print("\n" + "=" * 100)
    print("CORRECTLY RANKED EXAMPLES")
    print("=" * 100)

    for i, x in enumerate(correct_examples, 1):
        print(f"\nCORRECT EXAMPLE {i}")
        print("-" * 100)

        print("PROMPT:")
        print(short(x["prompt"]))

        print("\nCHOSEN:")
        print(short(x["chosen"]))

        print("\nREJECTED:")
        print(short(x["rejected"]))

        print("\nDataset scores:")
        print(
            f"chosen={x['score_chosen']}, "
            f"rejected={x['score_rejected']}"
        )

        print("\nReward model scores:")
        print(
            f"chosen={x['reward_chosen']:.4f}, "
            f"rejected={x['reward_rejected']:.4f}, "
            f"margin={x['margin']:.4f}"
        )

    print("\n" + "=" * 100)
    print("INCORRECTLY RANKED EXAMPLES")
    print("=" * 100)

    for i, x in enumerate(wrong_examples, 1):
        print(f"\nWRONG EXAMPLE {i}")
        print("-" * 100)

        print("PROMPT:")
        print(short(x["prompt"]))

        print("\nCHOSEN:")
        print(short(x["chosen"]))

        print("\nREJECTED:")
        print(short(x["rejected"]))

        print("\nDataset scores:")
        print(
            f"chosen={x['score_chosen']}, "
            f"rejected={x['score_rejected']}"
        )

        print("\nReward model scores:")
        print(
            f"chosen={x['reward_chosen']:.4f}, "
            f"rejected={x['reward_rejected']:.4f}, "
            f"margin={x['margin']:.4f}"
        )


if __name__ == "__main__":
    main()
