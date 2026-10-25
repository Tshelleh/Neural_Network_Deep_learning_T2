"""Hyper-parameter search that produces the table required in the Task 2 report.

    python experiments.py            full grid
    python experiments.py --quick    small grid, for a fast check

For every activation function it tries all combinations below and keeps the best one
(highest test accuracy; ties are broken by train accuracy, then fewer epochs, then fewer neurons).

Output (folder results/):
    all_runs.csv       every configuration that was tried
    best.csv           the best row per activation function (the report table)
    best_summary.txt   the same table + confusion matrices of the best runs
"""
import argparse
import csv
import itertools
import os
import time

from logic.metrics import format_confusion_matrix
from logic.pipeline import Pipeline, ACTIVATIONS

FULL_GRID = dict(
    architectures=[[2], [3], [5], [8], [3, 4], [4, 5], [5, 5], [8, 6]],   # neurons per hidden layer
    etas=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
    epochs=[100, 500, 1000, 5000],
    biases=[True, False],
)
QUICK_GRID = dict(architectures=[[3], [4, 5]], etas=[0.01, 0.1], epochs=[200, 1000], biases=[True])

FIELDS = ["activation", "train_accuracy", "test_accuracy", "lr", "epochs", "layers", "hidden_nodes", "bias"]


def rank_key(row):
    return (row["test_accuracy"], row["train_accuracy"], -row["epochs"], -sum(row["_arch"]))


def run_grid(pipe, grid):
    rows = []
    combos = list(itertools.product(ACTIVATIONS, grid["biases"], grid["architectures"],
                                    grid["etas"], grid["epochs"]))
    start = time.time()
    for n, (act, bias, arch, eta, epochs) in enumerate(combos, 1):
        res = pipe.train(len(arch), arch, eta, epochs, act, bias)
        rows.append({
            "activation": act,
            "train_accuracy": round(res.train_accuracy * 100, 2),
            "test_accuracy": round(res.test_accuracy * 100, 2),
            "lr": eta,
            "epochs": epochs,
            "layers": len(arch),
            "hidden_nodes": ",".join(map(str, arch)),
            "bias": bias,
            "_arch": arch,
            "_test_cm": res.test_cm,
        })
        if n % 50 == 0 or n == len(combos):
            print(f"  {n}/{len(combos)} runs  ({time.time() - start:.0f}s)", flush=True)
    return rows


def format_table(rows):
    header = ["Activation", "Train Acc %", "Test Acc %", "LR", "Epochs", "#Layers", "#HiddenNodes", "Bias"]
    body = [[r["activation"], f'{r["train_accuracy"]:.2f}', f'{r["test_accuracy"]:.2f}', str(r["lr"]),
             str(r["epochs"]), str(r["layers"]), r["hidden_nodes"], str(r["bias"])] for r in rows]
    widths = [max(len(x[i]) for x in [header] + body) for i in range(len(header))]
    line = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths))
    return "\n".join([line(header), line(["-" * w for w in widths])] + [line(b) for b in body])


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="use a small grid")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pipe = Pipeline()
    print("Running", "quick" if args.quick else "full", "grid...")
    rows = run_grid(pipe, QUICK_GRID if args.quick else FULL_GRID)

    best = [max((r for r in rows if r["activation"] == act), key=rank_key) for act in ACTIVATIONS]

    write_csv(os.path.join(args.out, "all_runs.csv"), rows)
    write_csv(os.path.join(args.out, "best.csv"), best)

    parts = ["Best result per activation function", format_table(best), ""]
    for r in best:
        parts += [f'{r["activation"]} - test confusion matrix',
                  format_confusion_matrix(r["_test_cm"], pipe.class_labels), ""]
    text = "\n".join(parts)
    with open(os.path.join(args.out, "best_summary.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()