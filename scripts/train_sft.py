import argparse
from pathlib import Path

import torch
import trl
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
DATASET_NAME = "trl-lib/Capybara"

QWEN3_TRAINING_TEMPLATE = str(
    Path(trl.__file__).resolve().parent
    / "chat_templates"
    / "qwen3_training.jinja"
)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/sft-smoke",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=256,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("SFT CONFIGURATION")
    print("=" * 80)
    print("Model:", MODEL_NAME)
    print("Dataset:", DATASET_NAME)
    print("Output:", args.output_dir)
    print("Max steps:", args.max_steps)
    print("Train samples:", args.train_samples)

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------
    dataset = load_dataset(DATASET_NAME)

    train_dataset = dataset["train"].shuffle(seed=42)
    eval_dataset = dataset["test"]

    # Fixed shuffle for reproducibility.
    dataset = dataset.shuffle(seed=42)

    if args.train_samples > 0:
        num_samples = min(args.train_samples, len(train_dataset))
        train_dataset = train_dataset.select(range(num_samples))

    print("Actual training samples:", len(train_dataset))
    print("Evaluation samples:", len(eval_dataset))

    # ---------------------------------------------------------
    # Training configuration
    # ---------------------------------------------------------
    training_args = SFTConfig(
        output_dir=args.output_dir,

        chat_template_path=QWEN3_TRAINING_TEMPLATE,

        model_init_kwargs={
            "dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        },

        assistant_only_loss=True,
        eos_token="<|im_end|>",

        max_length=2048,
        packing=False,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,

        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_steps=60,

        optim="adamw_torch_fused",

        bf16=True,
        fp16=False,

        gradient_checkpointing=True,

        num_train_epochs=1.0,

        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,

        eval_strategy="epoch",

        save_strategy="no",

        report_to="none",

        seed=42,
        data_seed=42,

        dataset_num_proc=4,
    )


    trainer = SFTTrainer(
        model=MODEL_NAME,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    train_result = trainer.train()

    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    eval_metrics = trainer.evaluate()

    for key, value in eval_metrics.items():
        print(f"{key}: {value}")

    print("\nSaving final SFT model...")
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
