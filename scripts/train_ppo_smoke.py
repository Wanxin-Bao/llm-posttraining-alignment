import torch
import argparse
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from trl.experimental.ppo import (
    PPOConfig,
    PPOTrainer,
)


TOKENIZER_PATH = "compat/qwen3-tokenizer-v4"

SFT_PATH = "outputs/qwen3-0.6b-sft"
REWARD_PATH = "outputs/qwen3-0.6b-reward"

DATASET_PATH = "data/ppo_prompts_512"

OUTPUT_DIR = "outputs/ppo-smoke"


def count_trainable(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--response-length",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--total-episodes",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/ppo-length-test",
    )

    return parser.parse_args()

def main():

    # ---------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------
    cli_args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        padding_side="left",
    )

    print("=" * 80)
    print("TOKENIZER")
    print("=" * 80)

    print("PAD:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("EOS:", tokenizer.eos_token, tokenizer.eos_token_id)

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset = load_from_disk(
        DATASET_PATH
    )

    # Tiny subsets are sufficient for the smoke test.
    train_dataset = dataset["train"].select(
        range(64)
    )

    eval_dataset = dataset["eval"].select(
        range(8)
    )

    print("\nTrain prompts:", len(train_dataset))
    print("Eval prompts:", len(eval_dataset))

    # ---------------------------------------------------------
    # Models
    # ---------------------------------------------------------

    model_kwargs = {
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
    }

    print("\nLoading policy...")

    policy = AutoModelForCausalLM.from_pretrained(
        SFT_PATH,
        **model_kwargs,
    )

    print("Loading reference policy...")

    reference = AutoModelForCausalLM.from_pretrained(
        SFT_PATH,
        **model_kwargs,
    )

    print("Loading reward model...")

    reward_model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            REWARD_PATH,
            num_labels=1,
            **model_kwargs,
        )
    )

    print("Loading value model...")

    value_model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            REWARD_PATH,
            num_labels=1,
            **model_kwargs,
        )
    )

    # Reference and reward model are frozen.
    for p in reference.parameters():
        p.requires_grad_(False)

    for p in reward_model.parameters():
        p.requires_grad_(False)

    print("\n" + "=" * 80)
    print("MODEL PARAMETERS")
    print("=" * 80)

    print(
        "Policy trainable:",
        f"{count_trainable(policy):,}",
    )

    print(
        "Reference trainable:",
        f"{count_trainable(reference):,}",
    )

    print(
        "Reward trainable:",
        f"{count_trainable(reward_model):,}",
    )

    print(
        "Value trainable:",
        f"{count_trainable(value_model):,}",
    )

    # ---------------------------------------------------------
    # PPO Configuration
    # ---------------------------------------------------------

    args = PPOConfig(
        output_dir=cli_args.output_dir,

        # -------------------------
        # Optimization
        # -------------------------

        learning_rate=1e-6,
        lr_scheduler_type="constant",

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,

        gradient_accumulation_steps=4,

        # One PPO pass over each rollout batch.
        num_ppo_epochs=1,
        num_mini_batches=1,

        # Only 16 sampled trajectories for smoke test.
        total_episodes=cli_args.total_episodes,

        # -------------------------
        # Rollout
        # -------------------------

        response_length=cli_args.response_length,
        temperature=0.7,

        # Generate until Qwen's EOS whenever possible.
        stop_token="eos",

        # We observed frequent truncated responses manually.
        missing_eos_penalty=1.0,

        # Process generation one prompt at a time
        # to reduce peak memory.
        local_rollout_forward_batch_size=1,

        # -------------------------
        # PPO
        # -------------------------

        kl_coef=0.05,
        kl_estimator="k1",

        cliprange=0.2,
        cliprange_value=0.2,

        vf_coef=0.1,

        gamma=1.0,
        lam=0.95,

        whiten_rewards=False,

        # -------------------------
        # Precision / memory
        # -------------------------

        bf16=True,
        fp16=False,

        gradient_checkpointing=True,

        # -------------------------
        # Logging
        # -------------------------

        logging_steps=1,
        logging_first_step=True,

        num_sample_generations=0,

        report_to="none",

        save_strategy="no",

        seed=42,
    )

    print("\n" + "=" * 80)
    print("PPO SMOKE CONFIG")
    print("=" * 80)

    print("Total episodes:", cli_args.total_episodes)
    print("Response length:", cli_args.response_length)
    print("LR:", args.learning_rate)
    print("KL coef:", args.kl_coef)
    print("Clip range:", args.cliprange)
    print("Gamma:", args.gamma)
    print("Lambda:", args.lam)

    # ---------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------

    print("\nCreating PPOTrainer...")

    trainer = PPOTrainer(
        args=args,
        processing_class=tokenizer,

        model=policy,
        ref_model=reference,

        reward_model=reward_model,
        value_model=value_model,

        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("\n" + "=" * 80)
    print("START PPO SMOKE TEST")
    print("=" * 80)

    trainer.train()

    print("\n" + "=" * 80)
    print("PPO SMOKE TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
