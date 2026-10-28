import ast
from pathlib import Path
import statistics


LOG_PATH = Path(
    "results/processed/day4_ppo_training.log"
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
                and "episode" in row
                and "objective/scores" in row
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

    episodes = len(rows) * 4

    eos = sum(
        float(row["val/num_eos_tokens"])
        for row in rows
    )

    eos_rate = eos / episodes

    print("=" * 72)
    print(name)
    print("=" * 72)

    print(f"Updates:              {len(rows)}")
    print(f"Episodes:             {episodes}")

    print(
        f"RM score:             "
        f"{mean(rows, 'objective/scores'):.4f}"
    )

    print(
        f"RLHF reward:          "
        f"{mean(rows, 'objective/rlhf_reward'):.4f}"
    )

    print(
        f"Sampled KL:           "
        f"{mean(rows, 'objective/kl'):.4f}"
    )

    print(
        f"PPO approx KL:        "
        f"{mean(rows, 'policy/approxkl_avg'):.6f}"
    )

    print(
        f"Policy ratio:         "
        f"{mean(rows, 'val/ratio'):.6f}"
    )

    print(
        f"Clip fraction:        "
        f"{100 * mean(rows, 'policy/clipfrac_avg'):.2f}%"
    )

    print(
        f"Value loss:           "
        f"{mean(rows, 'loss/value_avg'):.4f}"
    )

    print(
        f"Policy entropy:       "
        f"{mean(rows, 'policy/entropy_avg'):.4f}"
    )

    print(
        f"EOS count:            {eos:.0f}"
    )

    print(
        f"EOS rate:             "
        f"{100 * eos_rate:.2f}%"
    )

    print()


rows = load_rows(LOG_PATH)

print(f"\nTotal metric rows: {len(rows)}\n")

if len(rows) == 0:
    raise RuntimeError(
        "No PPO metric rows found."
    )


# Overall
summarize(
    "OVERALL",
    rows,
)


# Divide training into 4 equal quarters
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
