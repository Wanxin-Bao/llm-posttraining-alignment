import ast
import statistics
from pathlib import Path


FILES = {
    256: Path(
        "results/processed/day5_grpo_smoke.log"
    ),
    384: Path(
        "results/processed/day5_grpo_len384.log"
    ),
}


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

            if "reward_std" in row:
                rows.append(row)

    return rows


def mean(rows, key):
    values = [
        float(row[key])
        for row in rows
        if key in row
    ]

    return statistics.mean(values)


for length, path in FILES.items():

    rows = load_rows(path)

    print("=" * 70)
    print(
        f"Max completion length: {length}"
    )
    print("=" * 70)

    print(
        "Steps:",
        len(rows),
    )

    print(
        "Mean reward:",
        f"{mean(rows, 'reward'):.4f}",
    )

    print(
        "Mean reward std:",
        f"{mean(rows, 'reward_std'):.4f}",
    )

    print(
        "Zero-std fraction:",
        f"{100 * mean(rows, 'frac_reward_zero_std'):.2f}%",
    )

    print(
        "Mean completion length:",
        f"{mean(rows, 'completions/mean_length'):.2f}",
    )

    print(
        "Clipped completion ratio:",
        f"{100 * mean(rows, 'completions/clipped_ratio'):.2f}%",
    )

    print(
        "Mean grad norm:",
        f"{mean(rows, 'grad_norm'):.4f}",
    )

    print(
        "Max grad norm:",
        f"{max(float(r['grad_norm']) for r in rows):.4f}",
    )

    print(
        "Mean entropy:",
        f"{mean(rows, 'entropy'):.4f}",
    )

    print()
