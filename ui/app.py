"""Professional Dark-Mode Tkinter front-end for Neural Network Classifier.

Tabs
────
  ⚡ Training         — configure hyper-parameters and train a model
  🔬 Experiments      — run grid search with live progress & results table
  🎯 Classification   — classify a single penguin sample

All heavy computation runs in background threads so the UI stays responsive.
"""

import os
import sys
import csv
import queue
import time
import threading
import itertools
import tkinter as tk
from tkinter import ttk, messagebox

# ── project root ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from logic.pipeline import Pipeline, ACTIVATIONS
from logic.preprocessing import NUMERIC_COLS, ORIGIN_LOCATION
from logic.metrics import format_confusion_matrix

# ── constants ─────────────────────────────────────────────────────────
ORIGINS = [n for n in ORIGIN_LOCATION if n != "Other"]

FULL_GRID = dict(
    architectures=[[2], [3], [5], [8], [3, 4], [4, 5], [5, 5], [8, 6]],
    etas=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
    epochs=[100, 500, 1000, 5000],
    biases=[True, False],
)
QUICK_GRID = dict(
    architectures=[[3], [4, 5]],
    etas=[0.01, 0.1],
    epochs=[200, 1000],
    biases=[True],
)
CSV_FIELDS = [
    "activation", "train_accuracy", "test_accuracy",
    "lr", "epochs", "layers", "hidden_nodes", "bias",
]


def _rank(r):
    return (r["test_accuracy"], r["train_accuracy"],
            -r["epochs"], -sum(r["_arch"]))


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DARK COLOUR PALETTE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

C = dict(
    bg       = "#0d1117",   card     = "#161b22",   card2    = "#1c2128",
    hover    = "#21262d",   border   = "#30363d",   border2  = "#3d444d",
    text     = "#e6edf3",   text2    = "#8b949e",   text3    = "#484f58",
    accent   = "#8b5cf6",   accent2  = "#a78bfa",   accent3  = "#6d28d9",
    abg      = "#1e1332",
    green    = "#3fb950",   yellow   = "#d29922",   red      = "#f85149",
    blue     = "#58a6ff",   cyan     = "#39d0d8",
    row_alt  = "#12161c",   sel      = "#1f2a44",
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DARK THEME SETUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def apply_theme(root):
    """Apply a comprehensive dark theme to all ttk widgets."""
    s = ttk.Style(root)
    s.theme_use("clam")

    # ── base ──
    s.configure(".", background=C["bg"], foreground=C["text"],
                fieldbackground=C["card"], bordercolor=C["border"],
                troughcolor=C["card"], arrowcolor=C["text"],
                focuscolor=C["accent"])

    # ── notebook / tabs ──
    s.configure("TNotebook", background=C["bg"], borderwidth=0)
    s.configure("TNotebook.Tab", background=C["card"], foreground=C["text2"],
                padding=[24, 12], font=("Segoe UI", 10, "bold"))
    s.map("TNotebook.Tab",
          background=[("selected", C["bg"]), ("active", C["card2"])],
          foreground=[("selected", C["accent2"]), ("active", C["text"])])

    # ── frames ──
    s.configure("TFrame", background=C["bg"])

    # ── labels ──
    s.configure("TLabel",        background=C["bg"], foreground=C["text"],
                font=("Segoe UI", 10))
    s.configure("Card.TLabel",   background=C["card"], foreground=C["text"],
                font=("Segoe UI", 10))
    s.configure("CardDim.TLabel", background=C["card"], foreground=C["text2"],
                font=("Segoe UI", 10))
    s.configure("Sec.TLabel",    background=C["card"], foreground=C["accent2"],
                font=("Segoe UI", 11, "bold"))

    # ── entry / spinbox / combobox ──
    for w in ("TEntry", "TSpinbox", "TCombobox"):
        s.configure(w, fieldbackground=C["bg"], foreground=C["text"],
                    bordercolor=C["border"], insertcolor=C["text"],
                    selectbackground=C["accent3"], selectforeground="#fff",
                    padding=[10, 7])
        s.map(w, bordercolor=[("focus", C["accent"])],
              fieldbackground=[("readonly", C["bg"])])

    # ── buttons ──
    s.configure("TButton", background=C["accent"], foreground="#ffffff",
                padding=[24, 11], font=("Segoe UI", 10, "bold"), borderwidth=0)
    s.map("TButton",
          background=[("active", C["accent3"]), ("disabled", C["card2"])],
          foreground=[("disabled", C["text3"])])

    s.configure("Alt.TButton", background=C["card2"], foreground=C["text"],
                padding=[16, 9], font=("Segoe UI", 9, "bold"), borderwidth=0)
    s.map("Alt.TButton",
          background=[("active", C["hover"]), ("disabled", C["bg"])],
          foreground=[("disabled", C["text3"])])

    # ── check / radio ──
    for w in ("TCheckbutton", "TRadiobutton"):
        s.configure(w, background=C["card"], foreground=C["text"],
                    font=("Segoe UI", 10))
        s.map(w, background=[("active", C["card"])])
    
    # Make checkbutton indicator visible
    s.configure("TCheckbutton", indicatorcolor=C["accent2"], indicatorbackground=C["bg"])

    # ── progress bar ──
    s.configure("TProgressbar", background=C["accent"], troughcolor=C["card"],
                borderwidth=0, thickness=8)
    s.configure("Green.Horizontal.TProgressbar", background=C["green"])

    # ── treeview ──
    s.configure("Treeview", background=C["bg"], foreground=C["text"],
                fieldbackground=C["bg"], borderwidth=0, rowheight=32,
                font=("Segoe UI", 9))
    s.configure("Treeview.Heading", background=C["card"], foreground=C["text2"],
                borderwidth=0, font=("Segoe UI", 9, "bold"), padding=[8, 6])
    s.map("Treeview",
          background=[("selected", C["sel"])],
          foreground=[("selected", C["accent2"])])
    s.map("Treeview.Heading", background=[("active", C["card2"])])

    # ── scrollbar ──
    s.configure("Vertical.TScrollbar", background=C["card"], troughcolor=C["bg"],
                borderwidth=0, arrowsize=0, width=10)
    s.map("Vertical.TScrollbar", background=[("active", C["border"])])

    # ── separator ──
    s.configure("TSeparator", background=C["border"])

    # ── labelframe ──
    s.configure("TLabelframe", background=C["card"], bordercolor=C["border"])
    s.configure("TLabelframe.Label", background=C["card"], foreground=C["accent2"],
                font=("Segoe UI", 10, "bold"))

    # ── root + combobox popup ──
    root.configure(bg=C["bg"])
    root.option_add("*TCombobox*Listbox.background",       C["card"])
    root.option_add("*TCombobox*Listbox.foreground",       C["text"])
    root.option_add("*TCombobox*Listbox.selectBackground", C["accent"])
    root.option_add("*TCombobox*Listbox.selectForeground", "#fff")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB CLASSES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TrainingTab:
    """Training tab UI and logic."""

    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.h_var = tk.StringVar(value="2")
        self.n_var = tk.StringVar(value="4,5")
        self.eta_var = tk.StringVar(value="0.01")
        self.ep_var = tk.StringVar(value="1000")
        self.bias_var = tk.BooleanVar(value=True)
        self.act_var = tk.StringVar(value="Sigmoid")
        self._build()

    def _build(self):
        wrapper = tk.Frame(self.parent, bg=C["bg"])
        wrapper.pack(fill="both", expand=True)

        canvas = tk.Canvas(wrapper, bg=C["bg"], highlightthickness=0)
        vsb = ttk.Scrollbar(wrapper, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=C["bg"])
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw", tags="inner")
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        def _on_canvas_resize(event):
            canvas.itemconfig("inner", width=event.width)
        canvas.bind("<Configure>", _on_canvas_resize)

        def _wheel(event):
            canvas.yview_scroll(-int(event.delta / 120), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _wheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        cfg = self.app._card(inner, "\u2699\ufe0f  Configuration")

        self.app._row(cfg, "Number of hidden layers", self.h_var,
                      kind="spin", from_=1, to=10, width=11)
        self.app._row(cfg, "Neurons per layer (comma-sep)", self.n_var, width=15)
        self.app._row(cfg, "Learning rate (\u03b7)", self.eta_var, width=15)
        self.app._row(cfg, "Number of epochs", self.ep_var, width=15)

        cb = tk.Frame(cfg, bg=C["card"])
        cb.pack(fill="x", pady=4)
        tk.Checkbutton(cb, text="  Add bias", variable=self.bias_var, 
                       bg=C["card"], fg=C["text"], selectcolor=C["accent"],
                       activebackground=C["card"], activeforeground=C["text"],
                       font=("Segoe UI", 10)).pack(side="left")

        self.app._row(cfg, "Activation function", self.act_var,
                      kind="combo", values=list(ACTIVATIONS), width=13)

        btn_frame = tk.Frame(cfg, bg=C["card"])
        btn_frame.pack(pady=(14, 4))
        self.train_btn = ttk.Button(btn_frame, text="\u26a1  Train Model",
                                    command=self._on_train)
        self.train_btn.pack()

        self.train_prog = ttk.Progressbar(cfg, mode="indeterminate", length=300)
        self.train_prog.pack(pady=(8, 0))
        self.train_prog.pack_forget()

        res = self.app._card(inner, "\U0001f4ca  Results")

        self._train_status = tk.StringVar(value="Model not trained yet.")
        tk.Label(res, textvariable=self._train_status, font=("Segoe UI", 10),
                 fg=C["text2"], bg=C["card"]).pack(anchor="w", pady=(0, 10))

        acc_row = tk.Frame(res, bg=C["card"])
        acc_row.pack(fill="x", pady=4)

        self._train_acc = tk.StringVar(value="\u2014")
        self._test_acc = tk.StringVar(value="\u2014")

        for label, var, color in [("Train Accuracy", self._train_acc, C["blue"]),
                                   ("Test Accuracy", self._test_acc, C["green"])]:
            box = tk.Frame(acc_row, bg=C["hover"],
                           highlightbackground=C["border"], highlightthickness=1,
                           padx=18, pady=12)
            box.pack(side="left", padx=(0, 10), fill="x", expand=True)
            tk.Label(box, text=label, font=("Segoe UI", 9, "bold"),
                     fg=C["text2"], bg=C["hover"]).pack(anchor="w")
            tk.Label(box, textvariable=var, font=("Segoe UI", 28, "bold"),
                     fg=color, bg=C["hover"]).pack(anchor="w", pady=(4, 0))

        tk.Label(res, text="Test Confusion Matrix",
                 font=("Segoe UI", 10, "bold"), fg=C["text2"],
                 bg=C["card"]).pack(anchor="w", pady=(14, 6))
        self._cm = tk.Text(res, height=6, width=55, font=("Consolas", 10),
                           bg=C["bg"], fg=C["cyan"],
                           insertbackground=C["text"],
                           highlightbackground=C["border"], highlightthickness=1,
                           state="disabled", wrap="none", padx=12, pady=10)
        self._cm.pack(fill="x")

    def _read_inputs(self):
        try:
            h = int(self.h_var.get())
            neurons = [int(x) for x in self.n_var.get().replace(" ", "").split(",") if x]
        except ValueError:
            raise ValueError("Hidden layers / neurons must be integers (e.g. 4,5)")
        try:
            eta = float(self.eta_var.get())
            epochs = int(self.ep_var.get())
        except ValueError:
            raise ValueError("Learning rate must be a number, epochs an integer")
        return h, neurons, eta, epochs, self.act_var.get(), self.bias_var.get()

    def _on_train(self):
        try:
            params = self._read_inputs()
            self.app.pipe.validate_params(*params[:4])
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
            return

        self.train_btn.config(state="disabled")
        self.app.cls_btn.config(state="disabled")
        self._train_status.set("\u23f3  Training in progress\u2026")
        self._train_acc.set("\u2026")
        self._test_acc.set("\u2026")
        self.train_prog.pack(pady=(8, 0))
        self.train_prog.start(12)

        threading.Thread(target=self._train_worker, args=params, daemon=True).start()
        self.app.root.after(100, self._poll_train)

    def _train_worker(self, *params):
        try:
            self.app._tq.put(("ok", self.app.pipe.train(*params)))
        except Exception as exc:
            self.app._tq.put(("err", exc))

    def _poll_train(self):
        try:
            kind, data = self.app._tq.get_nowait()
        except queue.Empty:
            self.app.root.after(100, self._poll_train)
            return

        self.train_prog.stop()
        self.train_prog.pack_forget()

        if kind == "ok":
            self.app.model = data.model
            self._train_status.set("\u2705  Training complete!")
            self._train_acc.set(f"{data.train_accuracy:.2%}")
            self._test_acc.set(f"{data.test_accuracy:.2%}")
            self._cm.config(state="normal")
            self._cm.delete("1.0", "end")
            self._cm.insert("1.0",
                            format_confusion_matrix(data.test_cm, self.app.pipe.class_labels))
            self._cm.config(state="disabled")
            self.app.cls_btn.config(state="normal")
        else:
            self._train_status.set("\u274c  Training failed")
            messagebox.showerror("Error", str(data))

        self.train_btn.config(state="normal")


class ExperimentsTab:
    """Experiments tab UI and logic."""

    COLS = ("act", "train", "test", "lr", "ep", "layers", "neurons", "bias")
    HDRS = ("Activation", "Train %", "Test %", "LR", "Epochs",
            "Layers", "Neurons", "Bias")
    WIDTHS = (100, 85, 85, 75, 75, 65, 90, 60)

    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self._erows = []
        self._exp_running = False
        self._build()

    def _build(self):
        # Scrollable wrapper
        wrapper = tk.Frame(self.parent, bg=C["bg"])
        wrapper.pack(fill="both", expand=True)

        canvas = tk.Canvas(wrapper, bg=C["bg"], highlightthickness=0)
        vsb = ttk.Scrollbar(wrapper, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=C["bg"])
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw", tags="inner")
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        def _on_canvas_resize(event):
            canvas.itemconfig("inner", width=event.width)
        canvas.bind("<Configure>", _on_canvas_resize)

        def _wheel(event):
            canvas.yview_scroll(-int(event.delta / 120), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _wheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        ctrl = self.app._card(inner, "\U0001f52c  Hyper-Parameter Grid Search")

        g_row = tk.Frame(ctrl, bg=C["card"])
        g_row.pack(fill="x", pady=(0, 8))
        self._grid_var = tk.StringVar(value="quick")
        for val, txt, desc in [("quick", "\u26a1 Quick Grid", "(fast check)"),
                                ("full", "\U0001f525 Full Grid", "(comprehensive)")]:
            rf = tk.Frame(g_row, bg=C["card"])
            rf.pack(side="left", padx=(0, 20))
            ttk.Radiobutton(rf, text=f"  {txt}", variable=self._grid_var,
                            value=val).pack(side="left")
            tk.Label(rf, text=desc, font=("Segoe UI", 8),
                     fg=C["text3"], bg=C["card"]).pack(side="left", padx=(4, 0))

        b_row = tk.Frame(ctrl, bg=C["card"])
        b_row.pack(fill="x", pady=(4, 0))
        self._exp_btn = ttk.Button(b_row, text="\U0001f680  Run Grid Search",
                                   command=self._on_run)
        self._exp_btn.pack(side="left")
        self._exp_save = ttk.Button(b_row, text="\U0001f4be  Save CSV",
                                    style="Alt.TButton",
                                    command=self._on_save, state="disabled")
        self._exp_save.pack(side="left", padx=(10, 0))

        self._exp_pvar = tk.DoubleVar(value=0)
        self._exp_prog = ttk.Progressbar(ctrl, variable=self._exp_pvar,
                                          maximum=100, mode="determinate")
        self._exp_prog.pack(fill="x", pady=(10, 3))
        self._exp_status = tk.StringVar(value="Ready — choose a grid and press Run")
        tk.Label(ctrl, textvariable=self._exp_status, font=("Segoe UI", 9),
                 fg=C["text2"], bg=C["card"]).pack(anchor="w")

        tf = tk.Frame(inner, bg=C["bg"])
        tf.pack(fill="both", expand=True, padx=10, pady=(2, 4))

        self._tree = ttk.Treeview(tf, columns=self.COLS, show="headings", height=8)
        for c, h, w in zip(self.COLS, self.HDRS, self.WIDTHS):
            self._tree.heading(c, text=h,
                               command=lambda _c=c: self._sort_tree(_c, False))
            self._tree.column(c, width=w, anchor="center", minwidth=50)

        vsb = ttk.Scrollbar(tf, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self._tree.tag_configure("even", background=C["bg"])
        self._tree.tag_configure("odd", background=C["row_alt"])
        self._tree.tag_configure("best", background=C["abg"], foreground=C["accent2"])

        # Best results - simple text summary
        scard = self.app._card(inner, "\U0001f3c6  Best Results")
        self._best_text = tk.Text(scard, height=6, width=70,
                                   font=("Consolas", 9),
                                   bg=C["bg"], fg=C["cyan"],
                                   insertbackground=C["text"],
                                   highlightbackground=C["border"],
                                   highlightthickness=1,
                                   state="disabled", wrap="none",
                                   padx=10, pady=8)
        self._best_text.pack(fill="x")

    def _sort_tree(self, col, reverse):
        data = [(self._tree.set(iid, col), iid) for iid in self._tree.get_children()]
        try:
            data.sort(key=lambda t: float(t[0]), reverse=reverse)
        except ValueError:
            data.sort(key=lambda t: t[0].lower(), reverse=reverse)
        for idx, (_, iid) in enumerate(data):
            self._tree.move(iid, "", idx)
        self._tree.heading(col, command=lambda: self._sort_tree(col, not reverse))

    def _on_run(self):
        if self._exp_running:
            return
        self._exp_running = True
        self._erows.clear()
        self._tree.delete(*self._tree.get_children())
        self._exp_btn.config(state="disabled")
        self._exp_save.config(state="disabled")
        self._exp_pvar.set(0)
        self._exp_status.set("\u23f3  Starting grid search\u2026")

        grid = QUICK_GRID if self._grid_var.get() == "quick" else FULL_GRID
        threading.Thread(target=self._worker, args=(grid,), daemon=True).start()
        self.app.root.after(80, self._poll)

    def _worker(self, grid):
        combos = list(itertools.product(
            ACTIVATIONS, grid["biases"], grid["architectures"],
            grid["etas"], grid["epochs"]))
        total = len(combos)
        t0 = time.time()
        for n, (act, bias, arch, eta, ep) in enumerate(combos, 1):
            res = self.app.pipe.train(len(arch), arch, eta, ep, act, bias)
            row = dict(activation=act,
                       train_accuracy=round(res.train_accuracy * 100, 2),
                       test_accuracy=round(res.test_accuracy * 100, 2),
                       lr=eta, epochs=ep, layers=len(arch),
                       hidden_nodes=",".join(map(str, arch)),
                       bias=bias, _arch=arch, _test_cm=res.test_cm)
            self.app._eq.put(("row", row, n, total, time.time() - t0))
        self.app._eq.put(("done", None, total, total, time.time() - t0))

    def _poll(self):
        batch = 0
        while batch < 30:
            try:
                msg = self.app._eq.get_nowait()
            except queue.Empty:
                break
            batch += 1

            kind = msg[0]
            if kind == "row":
                _, row, n, total, elapsed = msg
                self._erows.append(row)
                tag = "even" if len(self._erows) % 2 == 0 else "odd"
                self._tree.insert("", "end", values=(
                    row["activation"], f'{row["train_accuracy"]:.2f}',
                    f'{row["test_accuracy"]:.2f}', str(row["lr"]), str(row["epochs"]),
                    str(row["layers"]), row["hidden_nodes"], str(row["bias"]),
                ), tags=(tag,))
                pct = n / total * 100
                self._exp_pvar.set(pct)
                self._exp_status.set(
                    f"Running: {n}/{total}  \u2502  {elapsed:.0f}s  \u2502  {pct:.0f}%")

            elif kind == "done":
                _, _, total, _, elapsed = msg
                self._exp_running = False
                self._exp_btn.config(state="normal")
                self._exp_save.config(state="normal")
                self._exp_pvar.set(100)
                self._exp_status.set(
                    f"\u2705  Complete \u2014 {total} configurations in {elapsed:.1f}s")
                self._show_summary()
                return

        if self._exp_running:
            self.app.root.after(60, self._poll)

    def _show_summary(self):
        best = {}
        for act in ACTIVATIONS:
            rows = [r for r in self._erows if r["activation"] == act]
            if rows:
                best[act] = max(rows, key=_rank)

        # Highlight best rows in main treeview
        for iid in self._tree.get_children():
            v = self._tree.item(iid, "values")
            act_name = v[0]
            if act_name in best:
                b = best[act_name]
                if (v[1] == f'{b["train_accuracy"]:.2f}' and
                    v[2] == f'{b["test_accuracy"]:.2f}' and
                    v[3] == str(b["lr"]) and
                    v[4] == str(b["epochs"]) and
                    v[6] == b["hidden_nodes"]):
                    self._tree.item(iid, tags=("best",))

        # Display best results as text
        lines = []
        for act, r in best.items():
            lines.append(
                f"{act:<10} Train:{r['train_accuracy']:>6.2f}%  Test:{r['test_accuracy']:>6.2f}%  "
                f"LR:{r['lr']}  Ep:{r['epochs']}  Neurons:{r['hidden_nodes']}  Bias:{r['bias']}"
            )
        
        self._best_text.config(state="normal")
        self._best_text.delete("1.0", "end")
        self._best_text.insert("1.0", "\n".join(lines))
        self._best_text.config(state="disabled")

    def _on_save(self):
        if not self._erows:
            return
        out = os.path.join(ROOT, "results")
        os.makedirs(out, exist_ok=True)

        _write_csv(os.path.join(out, "all_runs.csv"), self._erows)

        best_rows = []
        for act in ACTIVATIONS:
            rows = [r for r in self._erows if r["activation"] == act]
            if rows:
                best_rows.append(max(rows, key=_rank))
        _write_csv(os.path.join(out, "best.csv"), best_rows)

        parts = ["Best result per activation function"]
        header = ["Activation", "Train Acc %", "Test Acc %", "LR",
                  "Epochs", "#Layers", "#HiddenNodes", "Bias"]
        body = [[r["activation"], f'{r["train_accuracy"]:.2f}',
                 f'{r["test_accuracy"]:.2f}', str(r["lr"]),
                 str(r["epochs"]), str(r["layers"]),
                 r["hidden_nodes"], str(r["bias"])] for r in best_rows]
        widths = [max(len(x[i]) for x in [header] + body) for i in range(len(header))]
        fmt = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths))
        parts.append(fmt(header))
        parts.append(fmt(["-" * w for w in widths]))
        parts.extend(fmt(b) for b in body)
        parts.append("")
        for r in best_rows:
            parts.append(f'{r["activation"]} - test confusion matrix')
            parts.append(format_confusion_matrix(r["_test_cm"],
                                                  self.app.pipe.class_labels))
            parts.append("")
        with open(os.path.join(out, "best_summary.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write("\n".join(parts))

        messagebox.showinfo("Saved",
                            f"Results saved to:\n{out}\n\n"
                            "\u2022 all_runs.csv\n"
                            "\u2022 best.csv\n"
                            "\u2022 best_summary.txt")


class ClassificationTab:
    """Classification tab UI and logic."""

    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.sample_vars = {}
        self.origin_var = tk.StringVar(value=ORIGINS[0])
        self._build()

    def _build(self):
        inp = self.app._card(self.parent, "\U0001f4dd  Enter Sample Features")

        for col in NUMERIC_COLS:
            v = tk.StringVar()
            self.app._row(inp, col, v)
            self.sample_vars[col] = v

        self.app._row(inp, "OriginLocation", self.origin_var,
                      kind="combo", values=ORIGINS, width=14)

        bf = tk.Frame(inp, bg=C["card"])
        bf.pack(pady=(14, 4))
        self.cls_btn = ttk.Button(bf, text="\U0001f3af  Classify",
                                  command=self._on_classify, state="disabled")
        self.cls_btn.pack()

        tk.Label(inp, text="Train a model first (Training tab) to enable classification.",
                 font=("Segoe UI", 8), fg=C["text3"],
                 bg=C["card"]).pack(anchor="w", pady=(8, 0))

        pred = self.app._card(self.parent, "\U0001f427  Prediction")

        self._cls_result = tk.StringVar(value="No prediction yet")
        self._cls_label = tk.Label(pred, textvariable=self._cls_result,
                                   font=("Segoe UI", 20, "bold"),
                                   fg=C["text3"], bg=C["card"])
        self._cls_label.pack(pady=28)

    def _on_classify(self):
        if self.app.model is None:
            messagebox.showwarning("No model",
                                   "Please train a model first (Training tab).")
            return
        try:
            sample = {col: float(v.get()) for col, v in self.sample_vars.items()}
        except ValueError:
            messagebox.showerror("Invalid Input",
                                 "Enter a numeric value in every feature field.")
            return
        sample["OriginLocation"] = self.origin_var.get()
        _, name = self.app.pipe.classify(self.app.model, sample)
        self._cls_result.set(f"\U0001f427   {name}")
        self._cls_label.config(fg=C["cyan"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  APPLICATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class App:

    def __init__(self, root, pipeline):
        self.root = root
        self.pipe = pipeline
        self.model = None
        self._tq = queue.Queue()
        self._eq = queue.Queue()
        self.cls_btn = None

        root.title("Neural Network \u2014 Backpropagation Classifier")
        root.geometry("960x800")
        root.minsize(860, 700)
        apply_theme(root)
        self._build()
        self._center()

    # ────────────────────────────────────────────────────────────────
    #  helpers
    # ────────────────────────────────────────────────────────────────

    def _center(self):
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() - w) // 2
        y = (self.root.winfo_screenheight() - h) // 2
        self.root.geometry(f"+{x}+{y}")

    def _card(self, parent, title=None):
        """Dark card panel with subtle border."""
        f = tk.Frame(parent, bg=C["card"],
                     highlightbackground=C["border"], highlightthickness=1,
                     padx=20, pady=14)
        f.pack(fill="x", padx=10, pady=6)
        if title:
            tk.Label(f, text=title, font=("Segoe UI", 12, "bold"),
                     fg=C["accent2"], bg=C["card"]).pack(anchor="w", pady=(0, 10))
        return f

    def _row(self, parent, label_text, var, kind="entry", **kw):
        """Label + widget row inside a card."""
        r = tk.Frame(parent, bg=C["card"])
        r.pack(fill="x", pady=4)
        tk.Label(r, text=label_text, font=("Segoe UI", 10),
                 fg=C["text2"], bg=C["card"], width=30, anchor="w").pack(side="left")
        if kind == "entry":
            ttk.Entry(r, textvariable=var, width=kw.get("width", 15)).pack(side="left", padx=(4, 0))
        elif kind == "spin":
            ttk.Spinbox(r, textvariable=var, width=kw.get("width", 13),
                        from_=kw.get("from_", 1), to=kw.get("to", 99)).pack(side="left", padx=(4, 0))
        elif kind == "combo":
            ttk.Combobox(r, textvariable=var, values=kw.get("values", []),
                         state="readonly", width=kw.get("width", 13)).pack(side="left", padx=(4, 0))

    # ────────────────────────────────────────────────────────────────
    #  skeleton
    # ────────────────────────────────────────────────────────────────

    def _build(self):
        # ── header ──
        hdr = tk.Frame(self.root, bg=C["bg"])
        hdr.pack(fill="x", padx=28, pady=(20, 0))

        tk.Label(hdr, text="\U0001f9e0", font=("Segoe UI Emoji", 30),
                 bg=C["bg"]).pack(side="left", padx=(0, 14))

        tf = tk.Frame(hdr, bg=C["bg"])
        tf.pack(side="left")
        tk.Label(tf, text="Neural Network Classifier",
                 font=("Segoe UI", 20, "bold"), fg=C["text"],
                 bg=C["bg"]).pack(anchor="w")
        tk.Label(tf, text="Backpropagation  \u00b7  Penguin Species  \u00b7  Sigmoid / Tanh",
                 font=("Segoe UI", 9), fg=C["text2"],
                 bg=C["bg"]).pack(anchor="w", pady=(2, 0))

        # thin separator
        tk.Frame(self.root, bg=C["border"], height=1).pack(fill="x", padx=24, pady=(14, 0))

        # ── notebook ──
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=16, pady=12)

        f1 = ttk.Frame(nb)
        f2 = ttk.Frame(nb)
        f3 = ttk.Frame(nb)
        nb.add(f1, text="  \u26a1 Training  ")
        nb.add(f2, text="  \U0001f52c Experiments  ")
        nb.add(f3, text="  \U0001f3af Classification  ")

        self.training_tab = TrainingTab(f1, self)
        self.experiments_tab = ExperimentsTab(f2, self)
        self.classification_tab = ClassificationTab(f3, self)
        self.cls_btn = self.classification_tab.cls_btn

        # ── footer ──
        foot = tk.Frame(self.root, bg=C["card"], height=30)
        foot.pack(fill="x", side="bottom")
        foot.pack_propagate(False)
        tk.Label(foot, text="Task 2  \u00b7  Neural Networks & Deep Learning",
                 font=("Segoe UI", 8), fg=C["text3"],
                 bg=C["card"]).pack(pady=7)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run():
    root = tk.Tk()
    try:
        pipeline = Pipeline()
    except Exception as exc:
        messagebox.showerror("Cannot load data", str(exc))
        root.destroy()
        return
    App(root, pipeline)
    root.mainloop()



