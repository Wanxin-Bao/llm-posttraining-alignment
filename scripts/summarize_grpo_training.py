import ast
import statistics
from pathlib import Path


LOG_PATH = Path(
    "results/processed/day5_grpo_training.log"
)


def load_rows(path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        for line in f:
            line = line.strip()

            if not line.startswith("{"):
                continue

            try:
                row = ast.literal_eval(line)
            except Exception:
                continue

            if (
                isinstance(row, dict)
                and "reward" in row
                and "reward_std" in row
            ):
                rows.append(row)

    return rows


def mean(rows, key):
    values = [
        float(row[key])
        for row in rows
        if key in row
    ]

    if not values:
        return float("nan")

    return statistics.mean(values)


def summarize(name, rows):

    print("=" * 72)
    print(name)
    print("=" * 72)

    print(
        f"Steps:                       "
        f"{len(rows)}"
    )

    print(
        f"Mean reward:                 "
        f"{mean(rows, 'reward'):+.4f}"
    )

    print(
        f"Mean reward std:             "
        f"{mean(rows, 'reward_std'):.4f}"
    )

    print(
        f"Zero-std fraction:           "
        f"{100 * mean(rows, 'frac_reward_zero_std'):.2f}%"
    )

    print(
        f"Mean completion length:      "
        f"{mean(rows, 'completions/mean_length'):.2f}"
    )

    print(
        f"Clipped completion ratio:    "
        f"{100 * mean(rows, 'completions/clipped_ratio'):.2f}%"
    )

    terminated = [
        float(r["completions/mean_terminated_length"])
        for r in rows
        if "completions/mean_terminated_length" in r
    ]

    if terminated:
        print(
            f"Mean terminated length:      "
            f"{statistics.mean(terminated):.2f}"
        )

    print(
        f"Mean entropy:                "
        f"{mean(rows, 'entropy'):.4f}"
    )

    print(
        f"Mean grad norm:              "
        f"{mean(rows, 'grad_norm'):.4f}"
    )

    print(
        f"Max grad norm:               "
        f"{max(float(r['grad_norm']) for r in rows):.4f}"
    )

    print(
        f"Mean step time:              "
        f"{mean(rows, 'step_time'):.4f} s"
    )

    print(
        f"Mean clip region ratio:      "
        f"{100 * mean(rows, 'clip_ratio/region_mean'):.4f}%"
    )

    print(
        f"Mean learning rate:          "
        f"{mean(rows, 'learning_rate'):.10f}"
    )

    print()


rows = load_rows(LOG_PATH)

print()
print(
    "Total metric rows:",
    len(rows),
)
print()

if len(rows) == 0:
    raise RuntimeError(
        "No GRPO metric rows found."
    )


summarize(
    "OVERALL",
    rows,
)


n = len(rows)

q1 = rows[: n // 4]
q2 = rows[n // 4: n // 2]
q3 = rows[n // 2: 3 * n // 4]
q4 = rows[3 * n // 4:]


summarize(
    "Q1 - EARLY TRAINING",
    q1,
)

summarize(
    "Q2",
    q2,
)

summarize(
    "Q3",
    q3,
)

summarize(
    "Q4 - LATE TRAINING",
    q4,
)
