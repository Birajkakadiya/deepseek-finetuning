#!/usr/bin/env python3
"""Save training and validation loss graph from trainer_state.json."""
import argparse
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="outputs/t4")
    a = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = Path(a.output_dir)
    history = json.loads((folder / "trainer_state.json").read_text())["log_history"]
    train = [(x["step"], x["loss"]) for x in history if "step" in x and "loss" in x]
    val = [(x["step"], x["eval_loss"]) for x in history if "step" in x and "eval_loss" in x]
    if not train and not val:
        raise RuntimeError("No loss values found")
    plt.figure(figsize=(8, 4.5))
    if train: plt.plot(*zip(*train), marker="o", label="Train")
    if val: plt.plot(*zip(*val), marker="s", label="Validation")
    plt.xlabel("Optimizer step")
    plt.ylabel("Loss")
    plt.title("DeepSeek-Coder OpenCodeInstruct LoRA")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(folder / "loss.png", dpi=180)

if __name__ == "__main__":
    main()
