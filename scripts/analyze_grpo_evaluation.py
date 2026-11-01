import json
import statistics


PATH = (
    "results/processed/"
    "sft_vs_grpo_reward_evaluation.json"
)


with open(
    PATH,
    "r",
    encoding="utf-8",
) as f:
    data = json.load(f)


if isinstance(data, dict):
    for key in [
        "results",
        "examples",
        "records",
    ]:
        if key in data:
            data = data[key]
            break


margins = []

identical = 0
changed = 0

grpo_wins_changed = 0
sft_wins_changed = 0
ties_changed = 0


for row in data:

    sft_reward = float(
        row["sft_reward"]
    )

    grpo_reward = float(
        row["grpo_reward"]
    )

    margin = (
        grpo_reward
        - sft_reward
    )

    margins.append(margin)

    sft_response = row[
        "sft_response"
    ]

    grpo_response = row[
        "grpo_response"
    ]

    if sft_response == grpo_response:

        identical += 1

    else:

        changed += 1

        if margin > 0:
            grpo_wins_changed += 1

        elif margin < 0:
            sft_wins_changed += 1

        else:
            ties_changed += 1


positive = [
    x for x in margins
    if x > 0
]

negative = [
    x for x in margins
    if x < 0
]

zero = [
    x for x in margins
    if x == 0
]


print("=" * 70)
print("SFT vs GRPO EVALUATION ANALYSIS")
print("=" * 70)

print(
    f"Total prompts:          {len(data)}"
)

print(
    f"Identical responses:    "
    f"{identical} "
    f"({100 * identical / len(data):.2f}%)"
)

print(
    f"Changed responses:      "
    f"{changed} "
    f"({100 * changed / len(data):.2f}%)"
)

print()

print("Among changed responses:")

print(
    f"GRPO wins:              "
    f"{grpo_wins_changed}"
)

print(
    f"SFT wins:               "
    f"{sft_wins_changed}"
)

print(
    f"Reward ties:            "
    f"{ties_changed}"
)

non_ties = (
    grpo_wins_changed
    + sft_wins_changed
)

if non_ties > 0:

    print(
        f"GRPO non-tie win rate: "
        f"{100 * grpo_wins_changed / non_ties:.2f}%"
    )


print()
print("Reward margins:")

print(
    f"Positive margins:       {len(positive)}"
)

print(
    f"Negative margins:       {len(negative)}"
)

print(
    f"Zero margins:           {len(zero)}"
)

print(
    f"Mean margin:            "
    f"{statistics.mean(margins):+.6f}"
)

print(
    f"Median margin:          "
    f"{statistics.median(margins):+.6f}"
)

print(
    f"Mean absolute margin:   "
    f"{statistics.mean(abs(x) for x in margins):.6f}"
)

print(
    f"Mean positive margin:   "
    f"{statistics.mean(positive):+.6f}"
)

print(
    f"Mean negative margin:   "
    f"{statistics.mean(negative):+.6f}"
)
