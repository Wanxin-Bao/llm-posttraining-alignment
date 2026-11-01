import argparse
import torch

from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from trl import (
    GRPOConfig,
    GRPOTrainer,
)


TOKENIZER_PATH = "compat/qwen3-tokenizer-v4"

POLICY_PATH = "outputs/qwen3-0.6b-sft"
REWARD_MODEL_PATH = "outputs/qwen3-0.6b-reward"

DATASET_PATH = "data/grpo_prompts_512"


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-steps",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/qwen3-0.6b-grpo",
    )

    return parser.parse_args()


def count_trainable(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def main():

    cli_args = parse_args()

    # =========================================================
    # Tokenizers
    # =========================================================

    policy_tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        padding_side="left",
    )

    reward_tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        padding_side="right",
    )

    print("=" * 80)
    print("TOKENIZERS")
    print("=" * 80)

    print(
        "Policy PAD:",
        policy_tokenizer.pad_token,
        policy_tokenizer.pad_token_id,
    )

    print(
        "Policy EOS:",
        policy_tokenizer.eos_token,
        policy_tokenizer.eos_token_id,
    )

    print(
        "Reward PAD:",
        reward_tokenizer.pad_token,
        reward_tokenizer.pad_token_id,
    )

    # =========================================================
    # Dataset
    # =========================================================

    dataset = load_from_disk(
        DATASET_PATH
    )

    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]

    print()
    print("=" * 80)
    print("DATASET")
    print("=" * 80)

    print(
        "Train prompts:",
        len(train_dataset),
    )

    print(
        "Eval prompts:",
        len(eval_dataset),
    )

    print(
        "Columns:",
        train_dataset.column_names,
    )

    # =========================================================
    # Models
    # =========================================================

    model_kwargs = {
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
    }

    print()
    print("Loading SFT policy...")

    policy = AutoModelForCausalLM.from_pretrained(
        POLICY_PATH,
        **model_kwargs,
    )

    print("Loading Day-2 reward model...")

    reward_model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            REWARD_MODEL_PATH,
            num_labels=1,
            **model_kwargs,
        )
    )

    reward_model.config.pad_token_id = (
        reward_tokenizer.pad_token_id
    )

    print()
    print("=" * 80)
    print("MODEL PARAMETERS")
    print("=" * 80)

    print(
        "Policy trainable:",
        f"{count_trainable(policy):,}",
    )

    print(
        "Reward trainable:",
        f"{count_trainable(reward_model):,}",
    )

    # =========================================================
    # Freeze Reward Model
    # =========================================================

    for parameter in reward_model.parameters():
        parameter.requires_grad = False

    reward_model.eval()

    print(
        "Reward trainable after freeze:",
        f"{count_trainable(reward_model):,}",
    )

    # =========================================================
    # GRPO configuration
    # =========================================================

    args = GRPOConfig(

        output_dir=cli_args.output_dir,

        # -----------------------------------------------------
        # Training budget
        # -----------------------------------------------------

        max_steps=cli_args.max_steps,

        learning_rate=cli_args.learning_rate,

        lr_scheduler_type="linear",

        warmup_steps=0,

        # -----------------------------------------------------
        # Group construction
        # -----------------------------------------------------

        num_generations=4,

        per_device_train_batch_size=4,

        gradient_accumulation_steps=1,

        # -----------------------------------------------------
        # Generation
        # -----------------------------------------------------

        max_completion_length=(
            cli_args.max_completion_length
        ),

        temperature=0.7,

        top_p=0.95,

        # -----------------------------------------------------
        # Original GRPO baseline
        # -----------------------------------------------------

        loss_type="grpo",

        scale_rewards="group",

        num_iterations=1,

        epsilon=0.2,

        # No reference-policy KL for this baseline.
        beta=0.0,

        # -----------------------------------------------------
        # Truncated completions
        # -----------------------------------------------------

        mask_truncated_completions=False,

        # -----------------------------------------------------
        # Precision / optimization
        # -----------------------------------------------------

        bf16=True,

        fp16=False,

        gradient_checkpointing=True,

        max_grad_norm=1.0,

        # -----------------------------------------------------
        # Logging
        # -----------------------------------------------------

        logging_steps=1,

        logging_first_step=True,

        log_completions=False,

        report_to="none",

        # Save only the final model ourselves.
        save_strategy="no",

        seed=42,
    )

    print()
    print("=" * 80)
    print("MAIN GRPO CONFIG")
    print("=" * 80)

    print(
        "Max steps:",
        args.max_steps,
    )

    print(
        "Approx prompts:",
        args.max_steps,
    )

    print(
        "Approx sampled completions:",
        args.max_steps
        * args.num_generations,
    )

    print(
        "Learning rate:",
        args.learning_rate,
    )

    print(
        "Scheduler:",
        args.lr_scheduler_type,
    )

    print(
        "Group size:",
        args.num_generations,
    )

    print(
        "Train batch size:",
        args.per_device_train_batch_size,
    )

    print(
        "Gradient accumulation:",
        args.gradient_accumulation_steps,
    )

    print(
        "Max completion length:",
        args.max_completion_length,
    )

    print(
        "Loss type:",
        args.loss_type,
    )

    print(
        "Reward scaling:",
        args.scale_rewards,
    )

    print(
        "KL beta:",
        args.beta,
    )

    print(
        "Max grad norm:",
        args.max_grad_norm,
    )

    # =========================================================
    # Trainer
    # =========================================================

    print()
    print("Creating GRPOTrainer...")

    trainer = GRPOTrainer(
        model=policy,

        reward_funcs=reward_model,

        args=args,

        train_dataset=train_dataset,

        eval_dataset=eval_dataset,

        processing_class=policy_tokenizer,

        reward_processing_classes=reward_tokenizer,
    )

    print()
    print("=" * 80)
    print("START MAIN GRPO TRAINING")
    print("=" * 80)

    trainer.train()

    # =========================================================
    # Save final policy
    # =========================================================

    print()
    print("=" * 80)
    print("SAVING FINAL GRPO POLICY")
    print("=" * 80)

    trainer.save_model(
        cli_args.output_dir
    )

    policy_tokenizer.save_pretrained(
        cli_args.output_dir
    )

    print(
        "Saved final policy to:",
        cli_args.output_dir,
    )

    print()
    print("=" * 80)
    print("MAIN GRPO TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()