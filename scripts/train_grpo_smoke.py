import torch
import argparse
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

OUTPUT_DIR = "outputs/grpo-smoke"


def get_probe_parameter(model):
    """
    Pick a real transformer weight rather than an embedding scalar,
    and keep a small slice for before/after comparison.
    """
    for name, parameter in model.named_parameters():

        if "self_attn.q_proj.weight" in name:
            return name, parameter

    # Fallback if model naming differs.
    for name, parameter in model.named_parameters():

        if parameter.requires_grad and parameter.ndim >= 2:
            return name, parameter

    raise RuntimeError(
        "Could not find a trainable probe parameter."
    )

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/grpo-smoke",
    )

    return parser.parse_args()
def main():

    # =========================================================
    # Tokenizers
    # =========================================================
    cli_args = parse_args()

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
        "PAD:",
        policy_tokenizer.pad_token,
        policy_tokenizer.pad_token_id,
    )

    print(
        "EOS:",
        policy_tokenizer.eos_token,
        policy_tokenizer.eos_token_id,
    )

    # =========================================================
    # Dataset
    # =========================================================

    dataset = load_from_disk(
        DATASET_PATH
    )

    # 64 rows are more than enough for this 16-step smoke test.
    train_dataset = dataset["train"].select(
        range(64)
    )

    print()
    print("=" * 80)
    print("DATASET")
    print("=" * 80)

    print(
        "Train prompts:",
        len(train_dataset),
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

    # =========================================================
    # Record a policy weight before training
    # =========================================================

    probe_name, probe_parameter = (
        get_probe_parameter(policy)
    )

    probe_before = (
        probe_parameter
        .detach()
        .view(-1)[:8192]
        .float()
        .cpu()
        .clone()
    )

    print()
    print(
        "Probe parameter:",
        probe_name,
    )

    # =========================================================
    # GRPO config
    # =========================================================

    args = GRPOConfig(

        output_dir=cli_args.output_dir,
        # -----------------------------------------------------
        # Real learning
        # -----------------------------------------------------

        learning_rate=1e-6,

        max_steps=cli_args.max_steps,

        # -----------------------------------------------------
        # One group = 4 completions
        # -----------------------------------------------------

        num_generations=4,

        per_device_train_batch_size=4,

        gradient_accumulation_steps=1,

        # -----------------------------------------------------
        # Generation
        # -----------------------------------------------------

        max_completion_length=cli_args.max_completion_length,

        temperature=0.7,

        top_p=0.95,
        max_grad_norm=1.0,
        # -----------------------------------------------------
        # Original GRPO
        # -----------------------------------------------------

        loss_type="grpo",

        scale_rewards="group",

        num_iterations=1,

        epsilon=0.2,

        # No frozen reference KL in this baseline.
        beta=0.0,

        # -----------------------------------------------------
        # Precision / memory
        # -----------------------------------------------------

        bf16=True,

        fp16=False,

        gradient_checkpointing=True,

        # -----------------------------------------------------
        # Logging
        # -----------------------------------------------------

        logging_steps=1,

        logging_first_step=True,

        log_completions=False,

        report_to="none",

        save_strategy="no",

        seed=42,
    )

    print()
    print("=" * 80)
    print("GRPO SMOKE CONFIG")
    print("=" * 80)



    print(
        "Learning rate:",
        args.learning_rate,
    )

    print(
        "Group size:",
        args.num_generations,
    )

    print(
        "Batch size:",
        args.per_device_train_batch_size,
    )


    print(
        "Steps:",
        cli_args.max_steps,
    )

    print(
        "Max completion length:",
        cli_args.max_completion_length,
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

        processing_class=policy_tokenizer,

        reward_processing_classes=reward_tokenizer,
    )

    print()
    print("=" * 80)
    print("START GRPO SMOKE TRAINING")
    print("=" * 80)

    trainer.train()

    # =========================================================
    # Verify that weights actually changed
    # =========================================================

    probe_after = (
        probe_parameter
        .detach()
        .view(-1)[:8192]
        .float()
        .cpu()
    )

    delta = (
        probe_after
        - probe_before
    ).abs()

    changed_elements = int(
        (delta > 0).sum().item()
    )

    max_change = (
        delta.max().item()
    )

    mean_change = (
        delta.mean().item()
    )

    print()
    print("=" * 80)
    print("POLICY WEIGHT CHECK")
    print("=" * 80)

    print(
        "Probe parameter:",
        probe_name,
    )

    print(
        "Changed elements:",
        changed_elements,
        "/",
        len(probe_before),
    )

    print(
        "Max abs change:",
        max_change,
    )

    print(
        "Mean abs change:",
        mean_change,
    )

    print(
        "Policy updated:",
        changed_elements > 0,
    )

    print()
    print("=" * 80)
    print("GRPO SMOKE TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
