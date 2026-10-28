import json
import statistics
from pathlib import Path


DPO_PATH = Path(
    "results/processed/"
    "sft_vs_dpo_reward_evaluation.json"
)

PPO_PATH = Path(
    "results/processed/"
    "sft_vs_ppo_reward_evaluation.json"
)


def load_records(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    # Support either:
    #   [...]
    # or:
    #   {"results": [...]}
    if isinstance(data, dict):
        for key in [
            "results",
            "examples",
            "records",
        ]:
            if key in data:
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError(
            f"Could not find record list in {path}"
        )

    return data


def percentile(values, p):
    values = sorted(values)

    if not values:
        return float("nan")

    k = (
        (len(values) - 1)
        * p / 100.0
    )

    lower = int(k)
    upper = min(
        lower + 1,
        len(values) - 1,
    )

    weight = k - lower

    return (
        values[lower] * (1 - weight)
        + values[upper] * weight
    )


def analyze(
    name,
    records,
    aligned_prefix,
):
    margins = []
    changed_margins = []

    identical = 0

    for row in records:

        sft_reward = float(
            row["sft_reward"]
        )

        aligned_reward = float(
            row[f"{aligned_prefix}_reward"]
        )

        margin = (
            aligned_reward
            - sft_reward
        )

        margins.append(margin)

        sft_response = row.get(
            "sft_response"
        )

        aligned_response = row.get(
            f"{aligned_prefix}_response"
        )

        if (
            sft_response is not None
            and aligned_response is not None
        ):
            if (
                sft_response
                == aligned_response
            ):
                identical += 1
            else:
                changed_margins.append(
                    margin
                )

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

    print("=" * 72)
    print(name)
    print("=" * 72)

    print(
        f"Total prompts:            "
        f"{len(margins)}"
    )

    print(
        f"Positive margins:         "
        f"{len(positive)}"
    )

    print(
        f"Negative margins:         "
        f"{len(negative)}"
    )

    print(
        f"Zero margins:             "
        f"{len(zero)}"
    )

    print(
        f"Mean margin:              "
        f"{statistics.mean(margins):+.6f}"
    )

    print(
        f"Median margin:            "
        f"{statistics.median(margins):+.6f}"
    )

    print(
        f"Mean absolute margin:     "
        f"{statistics.mean(abs(x) for x in margins):.6f}"
    )

    if positive:
        print(
            f"Mean positive margin:     "
            f"{statistics.mean(positive):+.6f}"
        )

    if negative:
        print(
            f"Mean negative margin:     "
            f"{statistics.mean(negative):+.6f}"
        )

        print(
            f"Mean loss magnitude:      "
            f"{statistics.mean(abs(x) for x in negative):.6f}"
        )

    print()
    print("Margin percentiles:")

    for p in [
        10,
        25,
        50,
        75,
        90,
    ]:
        print(
            f"  P{p:02d}: "
            f"{percentile(margins, p):+.6f}"
        )

    if changed_margins:
        print()
        print(
            f"Identical responses:      "
            f"{identical}"
        )

        print(
            f"Changed responses:        "
            f"{len(changed_margins)}"
        )

        print(
            f"Changed mean margin:      "
            f"{statistics.mean(changed_margins):+.6f}"
        )

        print(
            f"Changed median margin:    "
            f"{statistics.median(changed_margins):+.6f}"
        )

        changed_positive = [
            x for x in changed_margins
            if x > 0
        ]

        changed_negative = [
            x for x in changed_margins
            if x < 0
        ]

        changed_zero = [
            x for x in changed_margins
            if x == 0
        ]

        print(
            f"Changed positive:         "
            f"{len(changed_positive)}"
        )

        print(
            f"Changed negative:         "
            f"{len(changed_negative)}"
        )

        print(
            f"Changed zero:             "
            f"{len(changed_zero)}"
        )

    print()

    return margins


dpo_records = load_records(
    DPO_PATH
)

ppo_records = load_records(
    PPO_PATH
)


dpo_margins = analyze(
    "DPO vs SFT",
    dpo_records,
    "dpo",
)

ppo_margins = analyze(
    "PPO vs SFT",
    ppo_records,
    "ppo",
)


print("=" * 72)
print("DIRECT COMPARISON")
print("=" * 72)

print(
    "DPO mean margin:",
    f"{statistics.mean(dpo_margins):+.6f}",
)

print(
    "PPO mean margin:",
    f"{statistics.mean(ppo_margins):+.6f}",
)

print(
    "Difference (DPO - PPO):",
    f"{statistics.mean(dpo_margins) - statistics.mean(ppo_margins):+.6f}",
)
