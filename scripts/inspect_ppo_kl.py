import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)


TOKENIZER_PATH = "compat/qwen3-tokenizer-v4"

POLICY_PATH = "outputs/qwen3-0.6b-sft"
REFERENCE_PATH = "outputs/qwen3-0.6b-sft"
REWARD_MODEL_PATH = "outputs/qwen3-0.6b-reward"

DATASET_NAME = "HuggingFaceH4/ultrafeedback_binarized"

DEVICE = "cuda"

NUM_PROMPTS = 3
MAX_NEW_TOKENS = 128

KL_COEF = 0.05


def get_response_logprobs(
    model,
    full_input_ids,
    prompt_length,
):
    """
    Compute log p(token_t | previous tokens)
    only for generated response tokens.
    """

    with torch.inference_mode():
        outputs = model(
            input_ids=full_input_ids
        )

    # logits[:, t] predicts input_ids[:, t+1]
    logits = outputs.logits[:, :-1, :].float()

    labels = full_input_ids[:, 1:]

    log_probs = F.log_softmax(
        logits,
        dim=-1,
    )

    token_logprobs = torch.gather(
        log_probs,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)

    # If prompt has P tokens:
    #
    # response starts at input_ids[P]
    #
    # that token is predicted by logits[P - 1].
    response_logprobs = token_logprobs[
        :,
        prompt_length - 1:
    ]

    return response_logprobs.squeeze(0)


def score_response(
    reward_model,
    tokenizer,
    prompt,
    response,
):
    messages = [
        {
            "role": "user",
            "content": prompt,
        },
        {
            "role": "assistant",
            "content": response,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(DEVICE)

    with torch.inference_mode():
        reward = reward_model(
            **inputs
        ).logits.squeeze()

    return reward.float().item()


def main():
    set_seed(42)

    # ---------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        padding_side="left",
    )

    # ---------------------------------------------------------
    # Policy
    # ---------------------------------------------------------

    print("Loading policy...")

    policy = AutoModelForCausalLM.from_pretrained(
        POLICY_PATH,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(DEVICE)

    policy.eval()

    # ---------------------------------------------------------
    # Frozen reference
    # ---------------------------------------------------------

    print("Loading reference policy...")

    reference = AutoModelForCausalLM.from_pretrained(
        REFERENCE_PATH,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(DEVICE)

    reference.eval()

    for p in reference.parameters():
        p.requires_grad_(False)

    # ---------------------------------------------------------
    # Reward model
    # ---------------------------------------------------------

    print("Loading reward model...")

    reward_model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            REWARD_MODEL_PATH,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(DEVICE)
    )

    reward_model.eval()

    for p in reward_model.parameters():
        p.requires_grad_(False)

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset = load_dataset(
        DATASET_NAME,
        split="test_prefs",
    )

    dataset = dataset.filter(
        lambda x:
        x["score_chosen"] > x["score_rejected"]
    )

    dataset = dataset.select(
        range(min(NUM_PROMPTS, len(dataset)))
    )

    # ---------------------------------------------------------
    # Rollout
    # ---------------------------------------------------------

    for i, sample in enumerate(dataset):

        prompt = sample["prompt"]

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        prompt_inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
        ).to(DEVICE)

        prompt_length = (
            prompt_inputs["input_ids"].shape[1]
        )

        # ---------------------------------------------
        # Online policy sampling
        # ---------------------------------------------

        with torch.inference_mode():
            generated = policy.generate(
                **prompt_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response_ids = generated[
            0,
            prompt_length:
        ]

        response = tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
        ).strip()

        # ---------------------------------------------
        # Same sampled trajectory scored by:
        #
        # current policy
        # frozen reference
        # ---------------------------------------------

        full_input_ids = generated

        policy_logps = get_response_logprobs(
            policy,
            full_input_ids,
            prompt_length,
        )

        reference_logps = get_response_logprobs(
            reference,
            full_input_ids,
            prompt_length,
        )

        assert (
            len(policy_logps)
            == len(reference_logps)
            == len(response_ids)
        )

        # sampled-token log-ratio
        log_ratio = (
            policy_logps
            - reference_logps
        )

        # KL-related non-score reward
        kl_reward = (
            -KL_COEF * log_ratio
        )

        sampled_kl = log_ratio.sum().item()

        total_kl_reward = (
            kl_reward.sum().item()
        )

        # ---------------------------------------------
        # Reward model score
        # ---------------------------------------------

        rm_score = score_response(
            reward_model,
            tokenizer,
            prompt,
            response,
        )

        rlhf_reward = (
            rm_score
            + total_kl_reward
        )

        # ---------------------------------------------
        # Print summary
        # ---------------------------------------------

        print("\n" + "=" * 90)
        print(f"ROLLOUT {i + 1}")
        print("=" * 90)

        print("\nPROMPT:")
        print(prompt)

        print("\nPOLICY RESPONSE:")
        print(response)

        print("\n" + "-" * 90)

        print(f"Generated tokens:      {len(response_ids)}")

        print(
            "Ended with EOS:       ",
            tokenizer.eos_token_id
            in response_ids.tolist(),
        )

        print(f"RM score:              {rm_score:+.6f}")
        print(f"Sampled KL estimate:   {sampled_kl:+.8f}")
        print(f"KL non-score reward:   {total_kl_reward:+.8f}")
        print(f"Final RLHF reward:     {rlhf_reward:+.6f}")

        # ---------------------------------------------
        # Token-level inspection
        # ---------------------------------------------

        print("\nTOKEN-LEVEL LOG PROBABILITIES")
        print("-" * 90)

        print(
            f"{'token':<20}"
            f"{'policy logp':>15}"
            f"{'ref logp':>15}"
            f"{'log-ratio':>15}"
            f"{'KL reward':>15}"
        )

        print("-" * 90)

        # Print first 12 tokens only to keep output readable.
        display_n = min(
            12,
            len(response_ids),
        )

        for j in range(display_n):

            token_id = response_ids[j].item()

            token_text = tokenizer.decode(
                [token_id]
            )

            token_text = repr(token_text)

            print(
                f"{token_text:<20}"
                f"{policy_logps[j].item():>15.6f}"
                f"{reference_logps[j].item():>15.6f}"
                f"{log_ratio[j].item():>15.8f}"
                f"{kl_reward[j].item():>15.8f}"
            )

        if len(response_ids) > display_n:
            print(
                f"... "
                f"{len(response_ids) - display_n} "
                "more tokens"
            )


if __name__ == "__main__":
    main()
