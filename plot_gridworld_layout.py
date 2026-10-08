"""Renders a static figure of the 6x6 gridworld layout (walls, gap/unsafe
cells, reach targets, reward-only target, start) for acc_paper.tex
Sec. VI.B. Separate from visualize.py's animated swarm videos -- this is
just the static environment diagram.
"""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

from sim import gridworld as env
from sim.visualize import _draw_static_grid


def main(out_path="figures/gridworld_layout.pdf"):
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    _draw_static_grid(ax)

    for (x, y) in env.WALL_CELLS:
        ax.text(x, y, "wall", ha="center", va="center", fontsize=6, color="white")
    for i, (x, y) in enumerate(env.UNSAFE_CELLS):
        ax.text(x, y, f"$s_{i+1}$", ha="center", va="center", fontsize=8)
    for (x, y) in env.REACH_TARGETS:
        ax.text(x, y, "T", ha="center", va="center", fontsize=8, weight="bold")
    rx, ry = env.REWARD_ONLY_TARGET
    ax.text(rx, ry, "R", ha="center", va="center", fontsize=8, weight="bold")
    sx, sy = env.START
    ax.text(sx, sy, "S", ha="center", va="center", fontsize=8, weight="bold",
            color="red")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    legend_handles = [
        Patch(facecolor="black", label="wall (impassable)"),
        Patch(facecolor="#ffcc66", alpha=0.7, label="gap cell $s_i$ (avoid)"),
        Patch(facecolor="#66cc66", alpha=0.7, label="reach target"),
        Patch(facecolor="#6699ff", alpha=0.7, label="reward-only target"),
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor="none",
               markeredgecolor="red", markersize=10, label="start"),
    ]
    ax.legend(handles=legend_handles, loc="center left",
              bbox_to_anchor=(1.02, 0.5), ncol=1, fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
