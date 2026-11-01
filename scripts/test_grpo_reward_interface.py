import torch

from datasets import Dataset, load_from_disk
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

PPO_DATASET_PATH = "data/ppo_prompts_512"

OUTPUT_DIR = "outputs/grpo-reward-interface-test"


def main():

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
    # Build tiny GRPO-format dataset
    # =========================================================

    ppo_dataset = load_from_disk(
        PPO_DATASET_PATH
    )

    prompts = []

    # Take several internal eval prompts.
    for example in ppo_dataset["eval"].select(
        range(8)
    ):

        prompt_text = policy_tokenizer.decode(
            example["input_ids"],
            skip_special_tokens=False,
        )

        prompts.append(prompt_text)

    grpo_dataset = Dataset.from_dict(
        {
            "prompt": prompts,
        }
    )

    print()
    print("=" * 80)
    print("TINY GRPO DATASET")
    print("=" * 80)

    print(
        "Rows:",
        len(grpo_dataset),
    )

    print(
        "Columns:",
        grpo_dataset.column_names,
    )

    print()
    print("Example prompt:")
    print("-" * 80)
    print(grpo_dataset[0]["prompt"])
    print("-" * 80)

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

    # Important for sequence classification.
    reward_model.config.pad_token_id = (
        reward_tokenizer.pad_token_id
    )

    # =========================================================
    # Record one policy parameter before dry-run
    # =========================================================

    first_parameter = next(
        policy.parameters()
    )

    before = (
        first_parameter
        .view(-1)[0]
        .detach()
        .float()
        .item()
    )

    print()
    print(
        "Policy parameter before:",
        before,
    )

    # =========================================================
    # GRPO dry-run config
    # =========================================================

    args = GRPOConfig(

        output_dir=OUTPUT_DIR,

        # -----------------------------------------------------
        # No real learning in Step 4.
        # -----------------------------------------------------

        learning_rate=0.0,

        max_steps=1,

        # -----------------------------------------------------
        # One GRPO group = 4 completions
        # -----------------------------------------------------

        num_generations=4,

        per_device_train_batch_size=4,

        gradient_accumulation_steps=1,

        # -----------------------------------------------------
        # Generation
        # -----------------------------------------------------

        max_completion_length=128,

        temperature=0.7,

        top_p=0.95,

        # -----------------------------------------------------
        # Explicit original GRPO behavior
        # -----------------------------------------------------

        loss_type="grpo",

        scale_rewards="group",

        num_iterations=1,

        epsilon=0.2,

        # No reference KL yet.
        # We will decide this separately later.
        beta=0.0,

        # -----------------------------------------------------
        # Precision
        # -----------------------------------------------------

        bf16=True,

        fp16=False,

        # Disable for this tiny interface test.
        gradient_checkpointing=False,

        # -----------------------------------------------------
        # Logging
        # -----------------------------------------------------

        logging_steps=1,

        logging_first_step=True,

        log_completions=True,

        num_completions_to_print=4,

        report_to="none",

        save_strategy="no",

        seed=42,
    )

    print()
    print("=" * 80)
    print("GRPO DRY-RUN CONFIG")
    print("=" * 80)

    print(
        "Learning rate:",
        args.learning_rate,
    )

    print(
        "Max steps:",
        args.max_steps,
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

    # =========================================================
    # GRPO Trainer
    # =========================================================

    print()
    print("Creating GRPOTrainer...")

    trainer = GRPOTrainer(
        model=policy,

        reward_funcs=reward_model,

        args=args,

        train_dataset=grpo_dataset,

        processing_class=policy_tokenizer,

        reward_processing_classes=reward_tokenizer,
    )

    print()
    print("=" * 80)
    print("START GRPO REWARD INTERFACE DRY RUN")
    print("=" * 80)

    trainer.train()

    # =========================================================
    # Confirm LR=0 caused no model update
    # =========================================================

    after = (
        first_parameter
        .view(-1)[0]
        .detach()
        .float()
        .item()
    )

    print()
    print("=" * 80)
    print("WEIGHT CHECK")
    print("=" * 80)

    print(
        "Before:",
        before,
    )

    print(
        "After: ",
        after,
    )

    print(
        "Changed:",
        before != after,
    )

    print()
    print("=" * 80)
    print(
        "GRPO REWARD INTERFACE TEST COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
