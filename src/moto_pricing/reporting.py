"""Matplotlib charts for the pipeline report. Kept deliberately small: one
function per chart, each saving a single PNG."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_model_comparison(comparison: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(comparison["model"], comparison["gini"], color=["#7f8c8d", "#2980b9", "#27ae60"])
    ax.set_ylabel("Normalised Gini (pure premium, out-of-time test)")
    ax.set_title("Model lift comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_actual_vs_expected(ave: pd.DataFrame, model_name: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = ave["decile"] + 1
    ax.plot(x, ave["actual"], marker="o", label="Actual")
    ax.plot(x, ave["expected"], marker="s", label="Expected")
    ax.set_xlabel("Predicted pure premium decile (1 = lowest risk)")
    ax.set_ylabel("Exposure-weighted mean")
    ax.set_title(f"Actual vs. expected — {model_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_categorical_relativities(relativities: pd.DataFrame, feature: str, out_path: Path) -> None:
    subset = relativities[relativities["feature"] == feature].sort_values("relativity")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(subset["level"].astype(str), subset["relativity"], color="#2980b9")
    ax.axvline(1.0, color="black", linewidth=0.8)
    ax.set_xlabel("Frequency relativity")
    ax.set_title(f"SHAP relativities — {feature}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_quote_funnel(funnel: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(funnel["stage"], funnel["count"], color="#8e44ad")
    ax.set_ylabel("Policies")
    ax.set_title("Quote-to-bind-to-renewal funnel")
    for i, v in enumerate(funnel["count"]):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
