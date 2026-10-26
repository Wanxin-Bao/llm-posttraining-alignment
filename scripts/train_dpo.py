import argparse

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import DPOConfig, DPOTrainer


MODEL_PATH = "outputs/qwen3-0.6b-sft"
DATASET_NAME = "HuggingFaceH4/ultrafeedback_binarized"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/dpo-smoke",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
    )


    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-7,
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


def preprocess_preference(example):
    """
    UltraFeedback chosen/rejected contain the complete conversation:

        user -> assistant

    DPO works most clearly with an explicit prompt plus two completions:

        prompt   = user conversation
        chosen   = preferred assistant completion
        rejected = dispreferred assistant completion
    """

    chosen_messages = example["chosen"]
    rejected_messages = example["rejected"]

    return {
        "prompt": chosen_messages[:-1],
        "chosen": [chosen_messages[-1]],
        "rejected": [rejected_messages[-1]],
    }


def prepare_split(dataset, seed=None):
    # Remove score ties exactly as in Day 2.
    dataset = dataset.filter(
        lambda x: x["score_chosen"] > x["score_rejected"]
    )

    dataset = dataset.map(
        preprocess_preference,
        remove_columns=dataset.column_names,
    )

    if seed is not None:
        dataset = dataset.shuffle(seed=seed)

    return dataset


def main():
    args = parse_args()

    print("=" * 80)
    print("DPO CONFIGURATION")
    print("=" * 80)

    print("Policy initialization :", MODEL_PATH)
    print("Reference policy      : frozen initial SFT policy")
    print("Dataset               :", DATASET_NAME)
    print("Output                :", args.output_dir)
    print("Max steps             :", args.max_steps)

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
                min(args.train_samples, len(train_dataset))
            )
        )

    if args.eval_samples > 0:
        eval_dataset = eval_dataset.select(
            range(
                min(args.eval_samples, len(eval_dataset))
            )
        )

    print("Training preference pairs :", len(train_dataset))
    print("Evaluation preference pairs:", len(eval_dataset))

    print("\nExample DPO sample:")
    print(train_dataset[0])

    # ---------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------------------------------------------------
    # DPO configuration
    # ---------------------------------------------------------

    training_args = DPOConfig(
        output_dir=args.output_dir,

        model_init_kwargs={
            "dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        },

        # Standard DPO
        beta=0.1,

        # Total prompt + completion length
        max_length=1024,

        # Optimization
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,

        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=1,

        optim="adamw_torch_fused",

        # Precision
        bf16=True,
        fp16=False,

        # Memory
        gradient_checkpointing=True,


        num_train_epochs=1.0,

        # Logging
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,

        eval_strategy="no",
        save_strategy="no",

        report_to="none",

        seed=42,
        data_seed=42,

        dataset_num_proc=4,
    )

    # ---------------------------------------------------------
    # DPO Trainer
    # ---------------------------------------------------------

    trainer = DPOTrainer(
        model=MODEL_PATH,

        # None means TRL automatically creates a frozen
        # reference corresponding to the initial SFT policy.
        ref_model=None,

        args=training_args,

        train_dataset=train_dataset,
        eval_dataset=eval_dataset,

        processing_class=tokenizer,
    )

    # ---------------------------------------------------------
    # Before training
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("PRE-DPO EVALUATION")
    print("=" * 80)

    before_metrics = trainer.evaluate()

    for key, value in before_metrics.items():
        print(f"{key}: {value}")

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("STARTING DPO TRAINING")
    print("=" * 80)

    train_result = trainer.train()

    print("\n" + "=" * 80)
    print("DPO TRAINING COMPLETE")
    print("=" * 80)

    for key, value in train_result.metrics.items():
        print(f"{key}: {value}")

    # ---------------------------------------------------------
    # After training
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("POST-DPO EVALUATION")
    print("=" * 80)

    after_metrics = trainer.evaluate()

    for key, value in after_metrics.items():
        print(f"{key}: {value}")

    print("\nSaving final DPO model...")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
