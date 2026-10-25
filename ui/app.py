"""Tkinter front end. Only widgets and event handling live here;
everything else (data, preprocessing, training, classification) is in logic/pipeline.py."""
import os
import sys
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from logic.pipeline import Pipeline, ACTIVATIONS, NUMERIC_FEATURES, ORIGINS
from logic.metrics import format_confusion_matrix


class App:
    def __init__(self, root, pipeline):
        self.root = root
        self.pipeline = pipeline
        self.model = None
        self.results = queue.Queue()
        root.title("Task 2 - Backpropagation Neural Network")
        root.resizable(False, False)
        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self):
        pad = {'padx': 8, 'pady': 4}

        # 1. user input
        box = ttk.LabelFrame(self.root, text="1. User Input")
        box.grid(row=0, column=0, sticky="ew", padx=10, pady=8)

        self.h_var = tk.StringVar(value="2")
        self.n_var = tk.StringVar(value="4,5")
        self.eta_var = tk.StringVar(value="0.01")
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
                     width=10).grid(row=5, column=1, **pad)

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
        for i, col in enumerate(NUMERIC_FEATURES):
            ttk.Label(cls, text=col).grid(row=i, column=0, sticky="w", **pad)
            v = tk.StringVar()
            ttk.Entry(cls, textvariable=v, width=12).grid(row=i, column=1, **pad)
            self.sample_vars[col] = v
        r = len(NUMERIC_FEATURES)
        ttk.Label(cls, text="OriginLocation").grid(row=r, column=0, sticky="w", **pad)
        self.origin_var = tk.StringVar(value=ORIGINS[0])
        ttk.Combobox(cls, textvariable=self.origin_var, values=ORIGINS, state="readonly",
                     width=10).grid(row=r, column=1, **pad)

        self.cls_btn = ttk.Button(cls, text="Classify", command=self.on_classify, state="disabled")
        self.cls_btn.grid(row=r + 1, column=0, columnspan=2, pady=6)
        self.cls_result = tk.StringVar(value="")
        ttk.Label(cls, textvariable=self.cls_result, font=("Segoe UI", 11, "bold")).grid(
            row=r + 2, column=0, columnspan=2, **pad)

    # ---------- training ----------
    def read_inputs(self):
        """Text boxes -> numbers. Range checks are done by the logic layer."""
        try:
            h = int(self.h_var.get())
            neurons = [int(p) for p in self.n_var.get().replace(" ", "").split(",") if p != ""]
        except ValueError:
            raise ValueError("Hidden layers and neurons must be integers, e.g. neurons: 4,5")
        try:
            eta = float(self.eta_var.get())
            epochs = int(self.ep_var.get())
        except ValueError:
            raise ValueError("Learning rate must be a number and epochs an integer")
        return h, neurons, eta, epochs, self.act_var.get(), self.bias_var.get()

    def on_train(self):
        try:
            params = self.read_inputs()
            self.pipeline.validate_params(*params[:4])
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))
            return
        self.train_btn.config(state="disabled")
        self.cls_btn.config(state="disabled")
        self.status.set("Training...")
        threading.Thread(target=self._train_worker, args=params, daemon=True).start()
        self.root.after(100, self._poll)

    def _train_worker(self, *params):
        # worker thread: never touch Tk widgets here, only put the outcome in the queue
        try:
            self.results.put(("ok", self.pipeline.train(*params)))
        except Exception as e:
            self.results.put(("error", e))

    def _poll(self):
        try:
            kind, payload = self.results.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll)
            return
        if kind == "ok":
            self.model = payload.model
            self.status.set(f"Train accuracy: {payload.train_accuracy:.2%}    "
                            f"Test accuracy: {payload.test_accuracy:.2%}")
            self.cm_label.config(text="Test confusion matrix\n" +
                                 format_confusion_matrix(payload.test_cm, self.pipeline.class_labels))
            self.cls_btn.config(state="normal")
        else:
            self.status.set("Training failed")
            messagebox.showerror("Error", str(payload))
        self.train_btn.config(state="normal")

    # ---------- classification ----------
    def on_classify(self):
        try:
            sample = {col: float(var.get()) for col, var in self.sample_vars.items()}
        except ValueError:
            messagebox.showerror("Invalid input", "Enter a numeric value in every feature field")
            return
        sample["OriginLocation"] = self.origin_var.get()
        _, name = self.pipeline.classify(self.model, sample)
        self.cls_result.set(f"Predicted: {name}")


def run():
    root = tk.Tk()
    try:
        pipeline = Pipeline()
    except Exception as e:
        messagebox.showerror("Cannot load data", str(e))
        root.destroy()
        return
    App(root, pipeline)
    root.mainloop()