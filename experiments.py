"""Hyper-parameter search that produces the table required in the Task 2 report.

    python experiments.py            full grid  (console output)
    python experiments.py --quick    small grid  (console output)
    python experiments.py --gui      full grid + results window
    python experiments.py --quick --gui

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
import sys
import time
import tkinter as tk
from tkinter import ttk

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from logic.metrics import format_confusion_matrix
from logic.pipeline import Pipeline, ACTIVATIONS

FULL_GRID = dict(
    architectures=[[2], [3], [5], [8], [3, 4], [4, 5], [5, 5], [8, 6]],
    etas=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
    epochs=[100, 500, 1000, 5000],
    biases=[True, False],
)
QUICK_GRID = dict(architectures=[[3], [4, 5]], etas=[0.01, 0.1], epochs=[200, 1000], biases=[True])

FIELDS = ["activation", "train_accuracy", "test_accuracy", "lr", "epochs", "layers", "hidden_nodes", "bias"]

# ── Dark colour palette (shared with ui/app.py) ──
C = dict(
    bg="#0d1117", card="#161b22", card2="#1c2128", hover="#21262d",
    border="#30363d", border2="#3d444d",
    text="#e6edf3", text2="#8b949e", text3="#484f58",
    accent="#8b5cf6", accent2="#a78bfa", accent3="#6d28d9", abg="#1e1332",
    green="#3fb950", yellow="#d29922", red="#f85149",
    blue="#58a6ff", cyan="#39d0d8",
    row_alt="#12161c", sel="#1f2a44",
)


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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GUI RESULTS WINDOW
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ResultsGUI:
    """Dark-themed GUI for displaying grid search results."""

    COLS = ("act", "train", "test", "lr", "ep", "layers", "neurons", "bias")
    HDRS = ("Activation", "Train %", "Test %", "LR", "Epochs", "Layers", "Neurons", "Bias")
    WIDTHS = (100, 85, 85, 75, 75, 65, 90, 60)

    def __init__(self, all_rows, best, class_labels):
        self.all_rows = all_rows
        self.best = best
        self.class_labels = class_labels
        self.root = tk.Tk()
        self._setup_window()
        self._apply_theme()
        self._build_ui()
        self._center()

    def _setup_window(self):
        self.root.title("Experiments — Grid Search Results")
        self.root.geometry("960x720")
        self.root.minsize(800, 600)

    def _apply_theme(self):
        """Apply dark theme to the results window."""
        s = ttk.Style(self.root)
        s.theme_use("clam")

        s.configure(".", background=C["bg"], foreground=C["text"],
                    fieldbackground=C["card"], bordercolor=C["border"],
                    troughcolor=C["card"], arrowcolor=C["text"], focuscolor=C["accent"])

        s.configure("TNotebook", background=C["bg"], borderwidth=0)
        s.configure("TNotebook.Tab", background=C["card"], foreground=C["text2"],
                    padding=[20, 10], font=("Segoe UI", 10, "bold"))
        s.map("TNotebook.Tab",
              background=[("selected", C["bg"]), ("active", C["card2"])],
              foreground=[("selected", C["accent2"]), ("active", C["text"])])

        s.configure("TFrame", background=C["bg"])
        s.configure("TLabel", background=C["bg"], foreground=C["text"], font=("Segoe UI", 10))

        s.configure("Treeview", background=C["bg"], foreground=C["text"],
                    fieldbackground=C["bg"], borderwidth=0, rowheight=30,
                    font=("Segoe UI", 9))
        s.configure("Treeview.Heading", background=C["card"], foreground=C["text2"],
                    borderwidth=0, font=("Segoe UI", 9, "bold"), padding=[6, 5])
        s.map("Treeview",
              background=[("selected", C["sel"])],
              foreground=[("selected", C["accent2"])])
        s.map("Treeview.Heading", background=[("active", C["card2"])])

        s.configure("Vertical.TScrollbar", background=C["card"], troughcolor=C["bg"],
                    borderwidth=0, arrowsize=0, width=10)
        s.map("Vertical.TScrollbar", background=[("active", C["border"])])

        self.root.configure(bg=C["bg"])

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=C["bg"])
        hdr.pack(fill="x", padx=24, pady=(18, 0))
        tk.Label(hdr, text="\U0001f52c", font=("Segoe UI Emoji", 26),
                 bg=C["bg"]).pack(side="left", padx=(0, 12))
        tf = tk.Frame(hdr, bg=C["bg"])
        tf.pack(side="left")
        tk.Label(tf, text="Grid Search Results",
                 font=("Segoe UI", 18, "bold"), fg=C["text"],
                 bg=C["bg"]).pack(anchor="w")
        tk.Label(tf, text=f"{len(self.all_rows)} configurations tested  \u00b7  "
                          f"{len(self.best)} activation functions",
                 font=("Segoe UI", 9), fg=C["text2"],
                 bg=C["bg"]).pack(anchor="w", pady=(2, 0))
        tk.Frame(self.root, bg=C["border"], height=1).pack(fill="x", padx=20, pady=(12, 0))

    def _build_best_tab(self, tab):
        best_tree = ttk.Treeview(tab, columns=self.COLS, show="headings", height=6)
        for c, h, w in zip(self.COLS, self.HDRS, self.WIDTHS):
            best_tree.heading(c, text=h)
            best_tree.column(c, width=w, anchor="center")
        best_tree.tag_configure("gold", background=C["abg"], foreground=C["accent2"])
        for r in self.best:
            best_tree.insert("", "end", values=(
                r["activation"], f'{r["train_accuracy"]:.2f}',
                f'{r["test_accuracy"]:.2f}', str(r["lr"]),
                str(r["epochs"]), str(r["layers"]),
                r["hidden_nodes"], str(r["bias"]),
            ), tags=("gold",))
        best_tree.pack(fill="x", padx=10, pady=(10, 6))

        cm_frame = tk.Frame(tab, bg=C["bg"])
        cm_frame.pack(fill="both", expand=True, padx=10, pady=4)
        tk.Label(cm_frame, text="\U0001f4ca  Confusion Matrices",
                 font=("Segoe UI", 12, "bold"), fg=C["accent2"],
                 bg=C["bg"]).pack(anchor="w", pady=(4, 8))

        cm_text = tk.Text(cm_frame, font=("Consolas", 10),
                          bg=C["card"], fg=C["cyan"],
                          insertbackground=C["text"],
                          highlightbackground=C["border"], highlightthickness=1,
                          state="normal", wrap="none", padx=14, pady=10)
        cm_text.pack(fill="both", expand=True)

        lines = []
        for r in self.best:
            lines.append(f"\u25b6  {r['activation']}")
            lines.append("")
            lines.append(format_confusion_matrix(r["_test_cm"], self.class_labels))
            lines.append("")
            lines.append("")
        cm_text.insert("1.0", "\n".join(lines))
        cm_text.config(state="disabled")

    def _build_all_runs_tab(self, tab):
        tree_f = tk.Frame(tab, bg=C["bg"])
        tree_f.pack(fill="both", expand=True, padx=10, pady=10)

        all_tree = ttk.Treeview(tree_f, columns=self.COLS, show="headings", height=20)
        for c, h, w in zip(self.COLS, self.HDRS, self.WIDTHS):
            all_tree.heading(c, text=h)
            all_tree.column(c, width=w, anchor="center")

        vsb = ttk.Scrollbar(tree_f, orient="vertical", command=all_tree.yview)
        all_tree.configure(yscrollcommand=vsb.set)
        all_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        all_tree.tag_configure("even", background=C["bg"])
        all_tree.tag_configure("odd", background=C["row_alt"])
        all_tree.tag_configure("best", background=C["abg"], foreground=C["accent2"])

        best_keys = set()
        for r in self.best:
            best_keys.add((r["activation"], f'{r["train_accuracy"]:.2f}',
                           f'{r["test_accuracy"]:.2f}', str(r["lr"]),
                           str(r["epochs"]), r["hidden_nodes"]))

        for i, r in enumerate(self.all_rows):
            vals = (r["activation"], f'{r["train_accuracy"]:.2f}',
                    f'{r["test_accuracy"]:.2f}', str(r["lr"]),
                    str(r["epochs"]), str(r["layers"]),
                    r["hidden_nodes"], str(r["bias"]))
            key = (vals[0], vals[1], vals[2], vals[3], vals[4], vals[6])
            tag = "best" if key in best_keys else ("even" if i % 2 == 0 else "odd")
            all_tree.insert("", "end", values=vals, tags=(tag,))

    def _build_footer(self):
        foot = tk.Frame(self.root, bg=C["card"], height=28)
        foot.pack(fill="x", side="bottom")
        foot.pack_propagate(False)
        tk.Label(foot, text="Task 2  \u00b7  Neural Networks & Deep Learning",
                 font=("Segoe UI", 8), fg=C["text3"],
                 bg=C["card"]).pack(pady=6)

    def _build_ui(self):
        self._build_header()

        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=14, pady=10)

        best_frame = ttk.Frame(nb)
        nb.add(best_frame, text="  \U0001f3c6 Best Results  ")
        self._build_best_tab(best_frame)

        all_frame = ttk.Frame(nb)
        nb.add(all_frame, text=f"  \U0001f4cb All {len(self.all_rows)} Runs  ")
        self._build_all_runs_tab(all_frame)

        self._build_footer()

    def _center(self):
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() - w) // 2
        y = (self.root.winfo_screenheight() - h) // 2
        self.root.geometry(f"+{x}+{y}")

    def show(self):
        self.root.mainloop()


def show_results_gui(all_rows, best, class_labels):
    """Open a Tk window that displays grid-search results in a dark-themed UI."""
    gui = ResultsGUI(all_rows, best, class_labels)
    gui.show()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="use a small grid")
    parser.add_argument("--gui", action="store_true",
                        help="show results in a dark-themed GUI window")
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

    # ── GUI mode ──
    if args.gui:
        print("\nOpening results window...")
        show_results_gui(rows, best, pipe.class_labels)


if __name__ == "__main__":
    main()
