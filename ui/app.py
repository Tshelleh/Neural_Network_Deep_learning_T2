import os
import sys
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ROOT for `logic.xxx` imports, ROOT/logic because some logic files use plain `import activations`
for p in (ROOT, os.path.join(ROOT, 'logic')):
    if p not in sys.path:
        sys.path.insert(0, p)

from logic.data_loader import load_data, split_data
from logic.preprocessing import preprocessing
from logic.activations import sigmoid, tanh
from logic.neural_network import neural_network
from logic.metrics import confusion_matrix, accuracy

NUM_COLS = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass']
ORIGINS = ['Biscoe', 'Dream', 'Torgersen']
ACTIVATIONS = {'Sigmoid': sigmoid, 'Hyperbolic Tangent (tanh)': tanh}


def recover_norm_stats(raw_train, norm_train):
    """Recover (mean, std) used by preprocessing for each numeric column.

    normalized = (raw - mean) / std is affine, so a straight-line fit of raw vs
    normalized (on rows that were not missing) gives std as slope and mean as
    intercept. This keeps the UI correct even if preprocessing changes how it
    fills missing values.
    """
    mean, std = {}, {}
    for col in NUM_COLS:
        ok = raw_train[col].notna().to_numpy()
        raw = raw_train.loc[ok, col].to_numpy(dtype=float)
        norm = norm_train.loc[ok, col].to_numpy(dtype=float)
        slope, intercept = np.polyfit(norm, raw, 1)
        mean[col], std[col] = intercept, slope
    return mean, std


def format_cm(cm, names):
    labels = [str(names.get(i, f"Class {i}")) for i in range(cm.shape[0])]
    w = max(len(l) for l in labels) + 2
    head = "Actual \\ Pred".ljust(w + 8) + "".join(l.rjust(w) for l in labels)
    rows = [head]
    for i, l in enumerate(labels):
        rows.append(l.ljust(w + 8) + "".join(str(v).rjust(w) for v in cm[i]))
    return "\n".join(rows)


class App:
    def __init__(self, root):
        self.root = root
        root.title("Task 2 - Backpropagation Neural Network")
        root.resizable(False, False)

        self.model = None
        self.results = queue.Queue()
        self._load_data()
        self._build_ui()

    # ---------- data ----------
    def _load_data(self):
        df = load_data(os.path.join(ROOT, 'data', 'penguins.csv'))
        train, test = split_data(df)
        raw_train = train.copy()          # preprocessing modifies train in place

        (self.X_train, self.y_train, self.X_test, self.y_test,
         target_encoder, self.origin_encode) = preprocessing(train, test)

        self.mean, self.std = recover_norm_stats(raw_train, self.X_train)

        # map one-hot column index -> original species name
        self.class_names = {}
        try:
            names = list(df['Species'].unique())
            enc = target_encoder.transform(pd.DataFrame({'Species': names}))
            for name, idx in zip(names, np.argmax(enc.values, axis=1)):
                self.class_names[int(idx)] = str(name)
        except Exception:
            self.class_names = {}

    # ---------- UI ----------
    def _build_ui(self):
        pad = {'padx': 8, 'pady': 4}

        # 1. user input
        box = ttk.LabelFrame(self.root, text="1. User Input")
        box.grid(row=0, column=0, sticky="ew", padx=10, pady=8)

        self.h_var = tk.StringVar(value="2")
        self.n_var = tk.StringVar(value="4,5")
        self.eta_var = tk.StringVar(value="0.5")
        self.ep_var = tk.StringVar(value="1000")
        self.bias_var = tk.BooleanVar(value=True)
        self.act_var = tk.StringVar(value="Sigmoid")

        ttk.Label(box, text="Number of hidden layers").grid(row=0, column=0, sticky="w", **pad)
        ttk.Spinbox(box, from_=1, to=10, textvariable=self.h_var, width=10).grid(row=0, column=1, **pad)

        ttk.Label(box, text="Neurons in each hidden layer (comma separated)").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(box, textvariable=self.n_var, width=12).grid(row=1, column=1, **pad)

        ttk.Label(box, text="Learning rate (eta)").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(box, textvariable=self.eta_var, width=12).grid(row=2, column=1, **pad)

        ttk.Label(box, text="Number of epochs (m)").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(box, textvariable=self.ep_var, width=12).grid(row=3, column=1, **pad)

        ttk.Checkbutton(box, text="Add bias", variable=self.bias_var).grid(row=4, column=0, sticky="w", **pad)

        ttk.Label(box, text="Activation function").grid(row=5, column=0, sticky="w", **pad)
        ttk.Combobox(box, textvariable=self.act_var, values=list(ACTIVATIONS), state="readonly",
                     width=24).grid(row=5, column=1, **pad)

        self.train_btn = ttk.Button(box, text="Train", command=self.on_train)
        self.train_btn.grid(row=6, column=0, columnspan=2, pady=8)

        # results
        res = ttk.LabelFrame(self.root, text="Results")
        res.grid(row=1, column=0, sticky="ew", padx=10, pady=8)
        self.status = tk.StringVar(value="Not trained yet")
        ttk.Label(res, textvariable=self.status).grid(row=0, column=0, sticky="w", **pad)
        self.cm_label = tk.Label(res, text="", font=("Consolas", 10), justify="left", anchor="w")
        self.cm_label.grid(row=1, column=0, sticky="w", **pad)

        # 3. classification
        cls = ttk.LabelFrame(self.root, text="3. Classification (single sample)")
        cls.grid(row=2, column=0, sticky="ew", padx=10, pady=8)
        self.sample_vars = {}
        for i, col in enumerate(NUM_COLS):
            ttk.Label(cls, text=col).grid(row=i, column=0, sticky="w", **pad)
            v = tk.StringVar()
            ttk.Entry(cls, textvariable=v, width=12).grid(row=i, column=1, **pad)
            self.sample_vars[col] = v
        ttk.Label(cls, text="OriginLocation").grid(row=len(NUM_COLS), column=0, sticky="w", **pad)
        self.origin_var = tk.StringVar(value=ORIGINS[0])
        ttk.Combobox(cls, textvariable=self.origin_var, values=ORIGINS, state="readonly",
                     width=10).grid(row=len(NUM_COLS), column=1, **pad)

        self.cls_btn = ttk.Button(cls, text="Classify", command=self.on_classify, state="disabled")
        self.cls_btn.grid(row=len(NUM_COLS) + 1, column=0, columnspan=2, pady=6)
        self.cls_result = tk.StringVar(value="")
        ttk.Label(cls, textvariable=self.cls_result, font=("Segoe UI", 11, "bold")).grid(
            row=len(NUM_COLS) + 2, column=0, columnspan=2, **pad)

    # ---------- training ----------
    def read_params(self):
        try:
            h = int(self.h_var.get())
        except ValueError:
            raise ValueError("Number of hidden layers must be an integer")
        if h < 1:
            raise ValueError("Number of hidden layers must be at least 1")
        try:
            neurons = [int(p) for p in self.n_var.get().replace(" ", "").split(",") if p != ""]
        except ValueError:
            raise ValueError("Neurons must be integers separated by commas, e.g. 4,5")
        if len(neurons) != h or any(n < 1 for n in neurons):
            raise ValueError(f"Enter exactly {h} positive integer(s) for neurons")
        try:
            eta = float(self.eta_var.get())
            epochs = int(self.ep_var.get())
        except ValueError:
            raise ValueError("Eta must be a number and epochs an integer")
        if eta <= 0 or epochs < 1:
            raise ValueError("Eta must be > 0 and epochs >= 1")
        return h, neurons, eta, epochs, ACTIVATIONS[self.act_var.get()], self.bias_var.get()

    def on_train(self):
        try:
            params = self.read_params()
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))
            return
        self.train_btn.config(state="disabled")
        self.cls_btn.config(state="disabled")
        self.status.set("Training...")
        threading.Thread(target=self._train_worker, args=params, daemon=True).start()
        self.root.after(100, self._poll)

    def _train_worker(self, h, neurons, eta, epochs, act, bias):
        # runs in a worker thread: never touch Tk widgets here, only put into the queue
        try:
            nn = neural_network(h, neurons, eta, epochs, act, bias)
            nn.train(self.X_train, self.y_train)

            def evaluate(X, y):
                pred = np.argmax(nn.predict(X), axis=1)
                true = np.argmax(np.asarray(y.values if hasattr(y, 'values') else y), axis=1)
                cm = confusion_matrix(true, pred)
                return cm, accuracy(cm)

            _, acc_tr = evaluate(self.X_train, self.y_train)
            cm_te, acc_te = evaluate(self.X_test, self.y_test)
            self.results.put(("ok", nn, acc_tr, acc_te, cm_te))
        except Exception as e:
            self.results.put(("error", e))

    def _poll(self):
        try:
            msg = self.results.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll)
            return
        if msg[0] == "ok":
            _, nn, acc_tr, acc_te, cm_te = msg
            self.model = nn
            self.status.set(f"Train accuracy: {acc_tr:.2%}    Test accuracy: {acc_te:.2%}")
            self.cm_label.config(text="Test confusion matrix\n" + format_cm(cm_te, self.class_names))
            self.cls_btn.config(state="normal")
        else:
            self.status.set("Training failed")
            messagebox.showerror("Error", str(msg[1]))
        self.train_btn.config(state="normal")

    # ---------- classification ----------
    def classify_sample(self, raw_values, origin):
        """raw_values: dict col -> raw float. Returns (class index, class name)."""
        vals = {col: (raw_values[col] - self.mean[col]) / self.std[col] for col in NUM_COLS}
        vals['OriginLocation'] = self.origin_encode.get(origin, self.origin_encode['Other'])
        row = [vals[c] for c in self.X_train.columns]
        x = np.array([row], dtype=np.float64)
        scores, _ = self.model.forward(x)
        idx = int(np.argmax(scores, axis=1)[0])
        return idx, self.class_names.get(idx, f"Class {idx}")

    def on_classify(self):
        try:
            raw = {col: float(self.sample_vars[col].get()) for col in NUM_COLS}
        except ValueError:
            messagebox.showerror("Invalid input", "Enter a numeric value in every feature field")
            return
        _, name = self.classify_sample(raw, self.origin_var.get())
        self.cls_result.set(f"Predicted: {name}")


def run():
    root = tk.Tk()
    App(root)
    root.mainloop()