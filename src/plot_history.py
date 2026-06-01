"""
plot_history.py
===============
Plots training/validation loss and metrics from training_history.csv.

Usage:
    python src/plot_history.py --csv models/training_history.csv
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot(csv_path: str):
    df  = pd.read_csv(csv_path)
    out = Path(csv_path).parent / "training_curves.png"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("VisionAST Training History", fontsize=14)

    # Loss
    axes[0].plot(df["epoch"], df["train_loss"], label="Train")
    axes[0].plot(df["epoch"], df["val_loss"],   label="Val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("BCE+Dice Loss"); axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Dice
    axes[1].plot(df["epoch"], df["dice"], color="green")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Dice")
    axes[1].set_title("Validation Dice Score"); axes[1].grid(True, alpha=0.3)

    # IoU
    axes[2].plot(df["epoch"], df["iou"], color="orange")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("IoU")
    axes[2].set_title("Validation IoU"); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(out), dpi=150)
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    plot(args.csv)
