"""Figure-generating functions for the drug resistance experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.stats import mannwhitneyu

from .drug_resistance_experiment import (
    CONDITIONS,
    MUTATION_SWEEP_RATES,
    N_GENERATIONS,
    RESISTANCE_THRESHOLD,
    SWEEP_CONDITIONS,
    ReplicateResult,
)
from .fitness import FitnessEvaluator
from .phenotype import Population

OUTPUT_DIR = Path("figures")

CONDITION_LABELS: dict[str, str] = {
    "control": "No treatment",
    "trail": "TRAIL",
    "bap15": "b-AP15",
    "combo": "b-AP15 + TRAIL",
    "triple": "b-AP15 + TRAIL + IAPi",
}
CONDITION_COLORS: dict[str, str] = {
    "control": "#888888",
    "trail": "#2196F3",
    "bap15": "#FF9800",
    "combo": "#E91E63",
    "triple": "#9C27B0",
}
CELL_LINE_LABELS: dict[str, str] = {
    "hpv_neg": "HPV−",
    "hpv_pos": "HPV+",
}


def _landscape(condition: str, resolution: int = 100) -> np.ndarray:
    n_vals = np.linspace(0, 1, resolution, dtype=np.float32)
    d_vals = np.linspace(0, 1, resolution, dtype=np.float32)
    nn, dd = np.meshgrid(n_vals, d_vals)
    pop = Population(nn.ravel(), dd.ravel())
    survival = FitnessEvaluator().evaluate(condition, pop)
    return survival.reshape(resolution, resolution)


def _select(
    results: list[ReplicateResult],
    condition: str,
    cell_line: str,
) -> list[ReplicateResult]:
    return [r for r in results if r.condition == condition and r.cell_line == cell_line]


def figure1_fitness_landscapes(output_dir: Path = OUTPUT_DIR) -> None:
    """Survival probability heatmap over n×d phenotype space per condition."""
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), constrained_layout=True)
    norm = Normalize(vmin=0, vmax=1)

    for ax, condition in zip(axes, CONDITIONS):
        surface = _landscape(condition)
        im = ax.imshow(
            surface,
            origin="lower",
            extent=[0, 1, 0, 1],
            cmap="RdYlGn",
            norm=norm,
            aspect="auto",
        )
        ax.set_title(CONDITION_LABELS[condition], fontsize=10)
        ax.set_xlabel("NFκB drive (n)")
        if ax is axes[0]:
            ax.set_ylabel("DR5 sensitivity (d)")

    fig.colorbar(im, ax=axes.tolist(), label="Survival probability S", shrink=0.8)
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "fig1_fitness_landscapes.pdf", dpi=150)
    plt.close(fig)


def figure2_trajectories(
    results: list[ReplicateResult],
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Mean phenotype trajectories ± SD for the HPV-negative population."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    gens = np.arange(N_GENERATIONS)

    for condition in CONDITIONS:
        reps = _select(results, condition, "hpv_neg")
        color = CONDITION_COLORS[condition]
        label = CONDITION_LABELS[condition]

        for attr, ax in zip(("mean_n", "mean_d"), axes):
            matrix = np.array([[getattr(g, attr) for g in r.trajectory] for r in reps])
            mean = matrix.mean(axis=0)
            std = matrix.std(axis=0)
            ax.plot(gens, mean, color=color, label=label, lw=1.5)
            ax.fill_between(gens, mean - std, mean + std, alpha=0.15, color=color)

    for ax, ylabel in zip(axes, ("Mean NFκB drive (n)", "Mean DR5 sensitivity (d)")):
        ax.set_xlabel("Generation")
        ax.set_ylabel(ylabel)
        ax.axhline(RESISTANCE_THRESHOLD, ls=":", color="black", lw=0.8)

    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("HPV− population — evolutionary trajectories")
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "fig2_trajectories.pdf", dpi=150)
    plt.close(fig)


def figure3_population_snapshots(
    results: list[ReplicateResult],
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Population scatter overlaid on fitness landscape at generations 0, 50, 149."""
    focus_conditions = ["trail", "combo"]
    snapshot_gens = [0, 50, N_GENERATIONS - 1]
    norm = Normalize(vmin=0, vmax=1)

    fig, axes = plt.subplots(
        len(focus_conditions),
        len(snapshot_gens),
        figsize=(12, 8),
        constrained_layout=True,
    )

    landscapes = {c: _landscape(c) for c in focus_conditions}

    for row, condition in enumerate(focus_conditions):
        rep = next(
            r for r in results if r.condition == condition
            and r.cell_line == "hpv_neg" and r.replicate == 0
        )
        snaps = {s.generation: s for s in rep.snapshots}

        for col, gen in enumerate(snapshot_gens):
            ax = axes[row, col]
            ax.imshow(
                landscapes[condition],
                origin="lower",
                extent=[0, 1, 0, 1],
                cmap="RdYlGn",
                norm=norm,
                aspect="auto",
                alpha=0.55,
            )
            if gen in snaps:
                snap = snaps[gen]
                ax.scatter(snap.n, snap.d, s=2, alpha=0.35, color=CONDITION_COLORS[condition])

            ax.set_title(f"Gen {gen}", fontsize=9)
            ax.set_xlabel("n")
            if col == 0:
                ax.set_ylabel(f"{CONDITION_LABELS[condition]}\nd")

    fig.suptitle("Population snapshots overlaid on fitness landscape")
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "fig3_snapshots.pdf", dpi=150)
    plt.close(fig)


def figure4_resistance_time(
    results: list[ReplicateResult],
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Box plots of resistance emergence generation across replicates and cell lines."""
    cell_lines = list(CELL_LINE_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True, sharey=True)

    sentinel = N_GENERATIONS + 15

    for ax, cell_line in zip(axes, cell_lines):
        data = []
        for condition in CONDITIONS:
            times = np.array([
                r.resistance_generation if r.resistance_generation != float("inf") else sentinel
                for r in _select(results, condition, cell_line)
            ], dtype=float)
            data.append(times)

        bp = ax.boxplot(
            data,
            tick_labels=[CONDITION_LABELS[c] for c in CONDITIONS],
            patch_artist=True,
            notch=False,
            medianprops={"color": "black", "lw": 1.5},
        )
        for patch, condition in zip(bp["boxes"], CONDITIONS):
            patch.set_facecolor(CONDITION_COLORS[condition])
            patch.set_alpha(0.7)

        ax.axhline(N_GENERATIONS, ls="--", color="black", lw=0.8)
        ax.set_title(CELL_LINE_LABELS[cell_line])
        ax.tick_params(axis="x", rotation=20)
        if ax is axes[0]:
            ax.set_ylabel("Resistance emergence (generation)")

    trail_times = np.array([
        r.resistance_generation if r.resistance_generation != float("inf") else sentinel
        for r in _select(results, "trail", "hpv_neg")
    ])
    combo_times = np.array([
        r.resistance_generation if r.resistance_generation != float("inf") else sentinel
        for r in _select(results, "combo", "hpv_neg")
    ])
    _, p = mannwhitneyu(trail_times, combo_times, alternative="less")
    fig.suptitle(f"Resistance emergence  |  TRAIL vs combo (HPV−): p = {p:.3f}")
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "fig4_resistance_time.pdf", dpi=150)
    plt.close(fig)


def figure5_hpv_comparison(
    results: list[ReplicateResult],
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Mean fitness trajectory comparing HPV+ vs HPV- under TRAIL and combo."""
    focus_conditions = ["trail", "combo"]
    line_styles = {"hpv_neg": "-", "hpv_pos": "--"}
    gens = np.arange(N_GENERATIONS)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True, sharey=True)

    for ax, condition in zip(axes, focus_conditions):
        for cell_line, ls in line_styles.items():
            reps = _select(results, condition, cell_line)
            matrix = np.array([[g.mean_s for g in r.trajectory] for r in reps])
            mean = matrix.mean(axis=0)
            std = matrix.std(axis=0)
            color = CONDITION_COLORS[condition]
            ax.plot(gens, mean, ls=ls, color=color, label=CELL_LINE_LABELS[cell_line], lw=1.5)
            ax.fill_between(gens, mean - std, mean + std, alpha=0.12, color=color)

        ax.axhline(RESISTANCE_THRESHOLD, ls=":", color="black", lw=0.8)
        ax.set_title(CONDITION_LABELS[condition])
        ax.set_xlabel("Generation")
        ax.legend(fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Mean survival S")

    fig.suptitle("HPV status comparison — mean fitness trajectory")
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "fig5_hpv_comparison.pdf", dpi=150)
    plt.close(fig)


def figure6_genetic_diversity(
    results: list[ReplicateResult],
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Genetic diversity (std of n and d) over time for the HPV-negative population."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    gens = np.arange(N_GENERATIONS)

    for condition in CONDITIONS:
        reps = _select(results, condition, "hpv_neg")
        color = CONDITION_COLORS[condition]
        label = CONDITION_LABELS[condition]

        for attr, ax in zip(("std_n", "std_d"), axes):
            matrix = np.array([[getattr(g, attr) for g in r.trajectory] for r in reps])
            mean = matrix.mean(axis=0)
            std = matrix.std(axis=0)
            ax.plot(gens, mean, color=color, label=label, lw=1.5)
            ax.fill_between(gens, mean - std, mean + std, alpha=0.15, color=color)

    axes[0].set_ylabel("σ(n) — NFκB drive diversity")
    axes[1].set_ylabel("σ(d) — DR5 sensitivity diversity")
    for ax in axes:
        ax.set_xlabel("Generation")
        ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("HPV− population — genetic diversity over time")
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "fig6_genetic_diversity.pdf", dpi=150)
    plt.close(fig)


def figure7_mutation_sweep(
    sweep_results: dict[float, list[ReplicateResult]],
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Resistance emergence time across mutation rates for TRAIL and combo (HPV-)."""
    sentinel = N_GENERATIONS + 15
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True, sharey=True)

    for ax, condition in zip(axes, SWEEP_CONDITIONS):
        data = []
        for rate in MUTATION_SWEEP_RATES:
            reps = [r for r in sweep_results[rate] if r.condition == condition]
            times = np.array([
                r.resistance_generation if r.resistance_generation != float("inf") else sentinel
                for r in reps
            ], dtype=float)
            data.append(times)

        bp = ax.boxplot(
            data,
            tick_labels=[f"σ = {r}" for r in MUTATION_SWEEP_RATES],
            patch_artist=True,
            notch=False,
            medianprops={"color": "black", "lw": 1.5},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(CONDITION_COLORS[condition])
            patch.set_alpha(0.7)

        ax.axhline(N_GENERATIONS, ls="--", color="black", lw=0.8)
        ax.set_title(CONDITION_LABELS[condition])
        ax.set_xlabel("Mutation std (σ)")
        if ax is axes[0]:
            ax.set_ylabel("Resistance emergence (generation)")

    fig.suptitle("Mutation rate sensitivity — resistance emergence (HPV−)")
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "fig7_mutation_sweep.pdf", dpi=150)
    plt.close(fig)


def figure8_survival_trajectories(
    results: list[ReplicateResult],
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Mean survival S over generations for all conditions (HPV-negative population)."""
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    gens = np.arange(N_GENERATIONS)

    for condition in CONDITIONS:
        reps = _select(results, condition, "hpv_neg")
        color = CONDITION_COLORS[condition]
        label = CONDITION_LABELS[condition]
        matrix = np.array([[g.mean_s for g in r.trajectory] for r in reps])
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        ax.plot(gens, mean, color=color, label=label, lw=1.5)
        ax.fill_between(gens, mean - std, mean + std, alpha=0.15, color=color)

    ax.axhline(
        RESISTANCE_THRESHOLD, ls=":", color="black", lw=0.8,
        label=f"Resistance threshold (S = {RESISTANCE_THRESHOLD})",
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean survival S")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("HPV− population — mean survival trajectories (all conditions)")
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / "fig8_survival_trajectories.pdf", dpi=150)
    plt.close(fig)


def generate_mutation_figures(
    sweep_results: dict[float, list[ReplicateResult]],
    output_dir: Path = OUTPUT_DIR,
) -> None:
    print("  Figure 7 (mutation sweep)...", end=" ", flush=True)
    figure7_mutation_sweep(sweep_results, output_dir=output_dir)
    print("done")
    print(f"Saved to {output_dir}/")


def generate_all(results: list[ReplicateResult], output_dir: Path = OUTPUT_DIR) -> None:
    numbered_fns = [
        (1, figure1_fitness_landscapes),
        (2, figure2_trajectories),
        (3, figure3_population_snapshots),
        (4, figure4_resistance_time),
        (5, figure5_hpv_comparison),
        (6, figure6_genetic_diversity),
        (8, figure8_survival_trajectories),
    ]
    for i, fn in numbered_fns:
        print(f"  Figure {i}...", end=" ", flush=True)
        if i == 1:
            fn(output_dir=output_dir)
        else:
            fn(results, output_dir=output_dir)
        print("done")
    print(f"Saved to {output_dir}/")
