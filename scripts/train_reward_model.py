import argparse

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import RewardConfig, RewardTrainer


MODEL_PATH = "outputs/qwen3-0.6b-sft"
DATASET_NAME = "HuggingFaceH4/ultrafeedback_binarized"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/reward-model-smoke",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--train-samples",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--eval-samples",
        type=int,
        default=256,
    )

    return parser.parse_args()


def prepare_split(dataset, seed=None):
    # Keep only pairs with an explicit positive preference.
    dataset = dataset.filter(
        lambda x: x["score_chosen"] > x["score_rejected"]
    )

    # IMPORTANT:
    # chosen/rejected already contain the full user+assistant conversation.
    # Remove "prompt" so RewardTrainer does not concatenate it again.
    dataset = dataset.select_columns(
        ["chosen", "rejected"]
    )

    # Disable Qwen3 thinking scaffolding for this preference experiment.
    dataset = dataset.map(
        lambda _: {
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }
    )

    if seed is not None:
        dataset = dataset.shuffle(seed=seed)

    return dataset


def main():
    args = parse_args()

    print("=" * 80)
    print("REWARD MODEL CONFIGURATION")
    print("=" * 80)

    print("Model:", MODEL_PATH)
    print("Dataset:", DATASET_NAME)
    print("Output:", args.output_dir)
    print("Max steps:", args.max_steps)

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset = load_dataset(DATASET_NAME)

    train_dataset = prepare_split(
        dataset["train_prefs"],
        seed=42,
    )

    eval_dataset = prepare_split(
        dataset["test_prefs"],
    )

    if args.train_samples > 0:
        train_dataset = train_dataset.select(
            range(
                min(
                    args.train_samples,
                    len(train_dataset),
                )
            )
        )

    if args.eval_samples > 0:
        eval_dataset = eval_dataset.select(
            range(
                min(
                    args.eval_samples,
                    len(eval_dataset),
                )
            )
        )

    print(
        "Training pairs before length filtering:",
        len(train_dataset),
    )

    print(
        "Evaluation pairs before length filtering:",
        len(eval_dataset),
    )

    # ---------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------------------------------------------------
    # Reward-model configuration
    # ---------------------------------------------------------

    training_args = RewardConfig(
        output_dir=args.output_dir,

        model_init_kwargs={
            "dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        },

        # Sequence filtering
        max_length=1024,

        # Optimization
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,

        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_steps=190,

        optim="adamw_torch_fused",

        # Precision
        bf16=True,
        fp16=False,

        # Memory
        gradient_checkpointing=True,

        num_train_epochs=1.0,

        # Logging
        logging_strategy="steps",
        logging_steps=20,
        logging_first_step=True,

        eval_strategy="no",
        save_strategy="no",

        report_to="none",

        seed=42,
        data_seed=42,

        dataset_num_proc=4,
    )

    # ---------------------------------------------------------
    # Reward Trainer
    # ---------------------------------------------------------

    trainer = RewardTrainer(
        model=MODEL_PATH,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print(
        "Training pairs after length filtering:",
        len(trainer.train_dataset),
    )

    print(
        "Evaluation pairs after length filtering:",
        len(trainer.eval_dataset),
    )

    # ---------------------------------------------------------
    # Before-training baseline
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("PRE-TRAINING REWARD EVALUATION")
    print("=" * 80)

    before_metrics = trainer.evaluate()

    for key, value in before_metrics.items():
        print(f"{key}: {value}")

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("STARTING REWARD MODEL TRAINING")
    print("=" * 80)

    train_result = trainer.train()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    for key, value in train_result.metrics.items():
        print(f"{key}: {value}")

    # ---------------------------------------------------------
    # After-training evaluation
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("POST-TRAINING REWARD EVALUATION")
    print("=" * 80)

    after_metrics = trainer.evaluate()

    for key, value in after_metrics.items():
        print(f"{key}: {value}")
        
    print("\nSaving final reward model...")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
