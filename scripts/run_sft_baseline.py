import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "outputs/qwen3-0.6b-sft"

PROMPTS = [
    "Explain why the sky appears blue in simple terms.",
    "Write a Python function that returns the largest number in a list.",
    "Give me three practical tips for preparing for a technical interview.",
    "What is the difference between supervised learning and reinforcement learning?",
    "Write a polite email asking a professor for a meeting next week.",
]


def main():
    print(f"Loading tokenizer: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print(f"Loading model: {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    results = []

    for idx, prompt in enumerate(PROMPTS, start=1):
        messages = [
            {"role": "user", "content": prompt},
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
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output_ids[
            0,
            inputs["input_ids"].shape[1]:
        ]

        response = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        print("=" * 80)
        print(f"[{idx}] PROMPT")
        print(prompt)

        print("\nSFT RESPONSE")
        print(response)

        results.append(
            {
                "id": idx,
                "prompt": prompt,
                "response": response,
            }
        )

    output_path = Path(
        "results/processed/sft_baseline_clean.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
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

    print("=" * 80)
    print(
        f"Saved SFT generations to: {output_path}"
    )


if __name__ == "__main__":
    main()
