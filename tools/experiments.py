"""Task 3: run one member's 10 hyperparameter experiments and write their results table.

    uv run tools/experiments.py --member shima --parallel 2   # train all 10, two at a time (finished ones are skipped)
    uv run tools/experiments.py --member shima --only 3       # train just experiment 3
    uv run tools/experiments.py --member shima --table        # only rebuild the results table
    uv run tools/experiments.py --member shima --table --best-steps 800000   # retrain the best one for longer

Each experiment changes the baseline below. Results go to models/<member>/expNN/ and a
table of the numbers to models/<member>/results_table.md. Each member then writes their
observations in results/<Name>_Hyperparameter_experiment_results.md (never overwritten here).
"""

import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

STEPS = 200_000  # same training length for every experiment, so they are comparable

BASELINE = {"lr": 1e-4, "gamma": 0.99, "batch_size": 32, "eps_start": 1.0, "eps_end": 0.02, "eps_decay": 100_000}

MEMBERS = {
    # Shima: change one hyperparameter at a time from the baseline, to see each one's effect alone.
    "shima": {
        "name": "Shima_Serein",
        "experiments": [
            {},  # 1. baseline
            {"lr": 5e-4},  # 2. higher learning rate
            {"lr": 2.5e-5},  # 3. lower learning rate
            {"gamma": 0.95},  # 4. shorter-sighted agent
            {"gamma": 0.90},  # 5. very short-sighted agent
            {"batch_size": 64},  # 6. bigger batches
            {"eps_start": 0.5},  # 7. less exploration at the start
            {"eps_end": 0.10},  # 8. keeps exploring more at the end
            {"eps_decay": 30_000},  # 9. exploration ends early
            {"eps_decay": 250_000},  # 10. exploration ends late
        ],
    },
    # Gaddiel: change several hyperparameters together, to look for the best combination.
    "gaddiel": {
        "name": "Gaddiel_Irakoze",
        "experiments": [
            {"batch_size": 64, "eps_end": 0.01, "eps_decay": 150_000},  # 1.
            {"lr": 2e-4, "batch_size": 64},  # 2.
            {"lr": 3e-4, "batch_size": 128},  # 3.
            {"lr": 5e-5, "eps_end": 0.01},  # 4.
            {"lr": 1e-3},  # 5. very high learning rate
            {"lr": 2e-4, "gamma": 0.97},  # 6.
            {"gamma": 0.995, "eps_decay": 150_000},  # 7.
            {"eps_end": 0.05, "eps_decay": 50_000},  # 8.
            {"eps_start": 0.8, "eps_end": 0.01, "eps_decay": 200_000},  # 9.
            {"lr": 2e-4, "eps_end": 0.01, "eps_decay": 60_000},  # 10.
        ],
    },
}


def config(changes):
    return {**BASELINE, **changes}


def run_training(member, name, steps, changes):
    """Run train.py with the baseline plus these changes. Skipped if this run already finished."""
    if (Path("models") / member / name / "training_log.csv").exists():
        print(f"{name} already done, skipping")
        return
    c = config(changes)
    command = [sys.executable, "train.py", "--member", member, "--name", name, "--steps", str(steps),
               "--lr", str(c["lr"]), "--gamma", str(c["gamma"]), "--batch-size", str(c["batch_size"]),
               "--eps-start", str(c["eps_start"]), "--eps-end", str(c["eps_end"]), "--eps-decay", str(c["eps_decay"])]
    print("\n" + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def train_best(member, steps):
    """Train the experiment with the highest final reward again, for longer. This is the final agent."""
    results = []
    for number, changes in enumerate(MEMBERS[member]["experiments"], start=1):
        log = Path("models") / member / f"exp{number:02d}" / "training_log.csv"
        if log.exists():
            results.append((summarize(log)["final_avg"], number, changes))
    best_avg, number, changes = max(results, key=lambda r: r[0])
    print(f"Best experiment: exp{number:02d} (avg reward {best_avg:.1f}). Training it for {steps:,} steps.")
    run_training(member, "best", steps, changes)


def summarize(log_path):
    """Numbers used in the table, read from an experiment's training_log.csv."""
    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    rewards = [float(r["reward"]) for r in rows]
    first_win = next((int(r["total_steps"]) for r in rows if float(r["reward"]) > 0), None)
    return {
        "episodes": len(rows),
        "final_avg": sum(rewards[-10:]) / len(rewards[-10:]),
        "best": max(rewards),
        "first_win": first_win,
        "final_length": sum(int(r["episode_length"]) for r in rows[-10:]) / len(rows[-10:]),
    }


def write_table(member):
    info = MEMBERS[member]
    lines = [
        f"# Experiment results (generated): {info['name'].replace('_', ' ')}",
        "",
        f"Environment: ALE/Pong-v5. Every experiment trains for {STEPS:,} steps with the same seed.",
        "Epsilon decays linearly from start to end over the number of steps in the decay column.",
        "Reward is points won minus points lost in one game (-21 to +21).",
        "",
        "| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for number, changes in enumerate(info["experiments"], start=1):
        c = config(changes)
        log = Path("models") / member / f"exp{number:02d}" / "training_log.csv"
        if log.exists():
            s = summarize(log)
            win = f"first won a game at step {s['first_win']:,}" if s["first_win"] else "never won a game"
            observed = (f"Avg reward of last 10 games: {s['final_avg']:.1f} (best {s['best']:.0f}); {win}; "
                        f"games lasted {s['final_length']:.0f} steps at the end.")
        else:
            observed = "Not run yet."
        lines.append(f"| {number} | {c['lr']:g} | {c['gamma']} | {c['batch_size']} | {c['eps_start']} | "
                     f"{c['eps_end']} | {c['eps_decay']:,} | {observed} |")
    path = Path("models") / member / "results_table.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def main():
    p = argparse.ArgumentParser(description="Run a member's hyperparameter experiments")
    p.add_argument("--member", required=True, choices=list(MEMBERS))
    p.add_argument("--only", type=int, help="run only this experiment number (1-10)")
    p.add_argument("--table", action="store_true", help="only rebuild the results table")
    p.add_argument("--parallel", type=int, default=1, help="how many experiments to train at the same time")
    p.add_argument("--best-steps", type=int, help="afterwards, retrain the best experiment for this many steps")
    args = p.parse_args()
    os.chdir(Path(__file__).resolve().parent.parent)  # work from the repo root, where train.py and models/ are

    if not args.table:
        experiments = MEMBERS[args.member]["experiments"]
        numbers = [n for n in range(1, len(experiments) + 1) if args.only in (None, n)]
        # One training run leaves a GPU mostly idle, so a few can share it.
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            list(pool.map(lambda n: run_training(args.member, f"exp{n:02d}", STEPS, experiments[n - 1]), numbers))
    write_table(args.member)
    if args.best_steps:
        train_best(args.member, args.best_steps)


if __name__ == "__main__":
    main()
