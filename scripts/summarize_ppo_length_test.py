import ast
from pathlib import Path


FILES = {
    128: Path(
        "results/processed/day4_length_ablation/"
        "ppo_len128.log"
    ),
    256: Path(
        "results/processed/day4_length_ablation/"
        "ppo_len256.log"
    ),
}


def parse_log(path):
    rows = []

    if not path.exists():
        return rows

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:

        for line in f:
            line = line.strip()

            # PPO metric rows are printed as Python dictionaries.
            if not line.startswith("{"):
                continue

            try:
                row = ast.literal_eval(line)
            except Exception:
                continue

            if (
                isinstance(row, dict)
                and "val/num_eos_tokens" in row
                and "episode" in row
            ):
                rows.append(row)

    return rows


for response_length, path in FILES.items():

    print("=" * 70)
    print(f"Response length: {response_length}")
    print("=" * 70)

    rows = parse_log(path)

    if not rows:
        print("No PPO metric rows found.")
        print("Expected log:", path)
        print()
        continue

    # Each logged batch reports how many EOS responses occurred
    # in that rollout batch.
    eos_count = sum(
        float(row["val/num_eos_tokens"])
        for row in rows
    )

    # Final logged episode count gives total trajectories processed.
    episodes = int(
        float(rows[-1]["episode"])
    )

    eos_rate = (
        eos_count / episodes
        if episodes > 0
        else 0.0
    )

    mean_score = sum(
        float(row["objective/scores"])
        for row in rows
    ) / len(rows)

    mean_rlhf_reward = sum(
        float(row["objective/rlhf_reward"])
        for row in rows
    ) / len(rows)

    mean_kl = sum(
        float(row["objective/kl"])
        for row in rows
    ) / len(rows)

    mean_approx_kl = sum(
        float(row["policy/approxkl_avg"])
        for row in rows
    ) / len(rows)

    mean_value_loss = sum(
        float(row["loss/value_avg"])
        for row in rows
    ) / len(rows)

    mean_ratio = sum(
        float(row["val/ratio"])
        for row in rows
    ) / len(rows)

    mean_clipfrac = sum(
        float(row["policy/clipfrac_avg"])
        for row in rows
    ) / len(rows)

    print(f"Metric rows:               {len(rows)}")
    print(f"Episodes:                  {episodes}")
    print(f"EOS count:                 {eos_count:.0f}")
    print(f"EOS rate:                  {100 * eos_rate:.2f}%")
    print(f"Mean RM score:             {mean_score:.4f}")
    print(f"Mean RLHF reward:          {mean_rlhf_reward:.4f}")
    print(f"Mean sampled KL:           {mean_kl:.4f}")
    print(f"Mean PPO approx KL:        {mean_approx_kl:.6f}")
    print(f"Mean value loss:           {mean_value_loss:.4f}")
    print(f"Mean policy ratio:         {mean_ratio:.6f}")
    print(f"Mean policy clip fraction: {100 * mean_clipfrac:.2f}%")
    print()
