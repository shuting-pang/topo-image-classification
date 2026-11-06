"""Regenerate the README figures: python examples/visualize_betti.py.

Dependencies: numpy, pillow, matplotlib, gudhi
"""
from pathlib import Path

import gudhi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parents[1] / "img"
STAGES = (50, 110, 150)
BLUE, ORANGE = "#2463a5", "#d56827"


def compute_betti(image):
    """Count intervals alive at each t: birth <= t < death, including infinity."""
    complex_ = gudhi.CubicalComplex(top_dimensional_cells=image.astype(float))
    complex_.compute_persistence(homology_coeff_field=2, min_persistence=0)
    thresholds = np.arange(256)
    curves = np.empty((2, len(thresholds)), dtype=np.int64)
    for dimension in (0, 1):
        intervals = complex_.persistence_intervals_in_dimension(dimension)
        curves[dimension] = (
            (intervals[:, 0, None] <= thresholds)
            & (thresholds < intervals[:, 1, None])
        ).sum(axis=0)
    return thresholds, curves


def main():
    image = np.asarray(Image.open(OUT / "mri_slice.png").convert("L"))
    thresholds, curves = compute_betti(image)
    # Rows: beta_0, beta_1. Columns: shared intensity thresholds 0..255.
    np.save(OUT / "mri_betti_curves.npy", curves)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

    fig, axes = plt.subplots(1, 4, figsize=(13, 4.3))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Input: MRI slice", fontweight="bold", pad=12)
    axes[0].set_xlabel("256 x 256 pixels")
    for ax, letter, t in zip(axes[1:], "ABC", STAGES):
        # Actual included square pixels of the cubical complex, not a point graph.
        ax.imshow(image <= t, cmap=ListedColormap(["#eef2f6", BLUE]),
                  vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"{letter}  |  threshold t = {t}", fontweight="bold", pad=12)
        ax.set_xlabel(rf"$\beta_0$ = {curves[0, t]}     $\beta_1$ = {curves[1, t]}")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Intensity filtration: add every pixel with I(x) <= t",
                 fontsize=16, fontweight="bold", y=0.99)
    fig.text(0.5, 0.04, "Blue = included pixels. As t increases, the sets grow: A is contained in B, and B in C.",
             ha="center", color="#435267")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.16, top=0.82, wspace=0.12)
    fig.savefig(OUT / "filtration_process.png", dpi=160, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for dim, ax, color, title in zip(
        (0, 1), axes, (BLUE, ORANGE),
        (r"$\beta_0$: connected components", r"$\beta_1$: loops / holes"),
    ):
        ax.step(thresholds, curves[dim], where="post", color=color, linewidth=1.8)
        for letter, t in zip("ABC", STAGES):
            ax.axvline(t, color="#bac3d0", linestyle="--", linewidth=1)
            ax.scatter(t, curves[dim, t], color=color, s=34, zorder=3)
            ax.text(t + 3, 0.94, f"{letter}\n{t}", transform=ax.get_xaxis_transform(),
                    va="top", fontsize=10, color="#435267")
        ax.set(title=title, xlabel="Intensity threshold t", ylabel="Number of features",
               xlim=(0, 255), ylim=(0, curves[dim].max() * 1.2))
        ax.set_xticks([0, 50, 110, 150, 200, 255])
        ax.grid(axis="y", color="#e5eaf0")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Betti curves from the same MRI slice", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.025, "A, B and C match the filtration panels. Each point counts the structures present at that threshold.",
             ha="center", color="#435267", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(OUT / "betti_curve.png", dpi=160, facecolor="white")
    plt.close(fig)
    print(f"Saved figures and Betti curves to {OUT}")
    for t in STAGES:
        print(f"t={t}: beta_0={curves[0, t]}, beta_1={curves[1, t]}")


if __name__ == "__main__":
    main()
