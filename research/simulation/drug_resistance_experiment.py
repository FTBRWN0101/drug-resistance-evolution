"""Main simulation: evolutionary dynamics of drug resistance in HNSCC populations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fitness import FitnessEvaluator
from .phenotype import Population

POPULATION_SIZE: int = 500
N_GENERATIONS: int = 400
MUTATION_STD: float = 0.02
N_REPLICATES: int = 20
RESISTANCE_THRESHOLD: float = 0.85

CELL_LINES: dict[str, dict[str, float]] = {
    "hpv_neg": {"n_mean": 0.50, "d_mean": 0.40},
    "hpv_pos": {"n_mean": 0.55, "d_mean": 0.30},
}
CONDITIONS: list[str] = ["control", "trail", "bap15", "combo", "triple"]
MUTATION_SWEEP_RATES: list[float] = [0.01, 0.02, 0.05]
SWEEP_CONDITIONS: list[str] = ["trail", "combo"]

# Conditions and replicates for which full population snapshots are stored
_SNAPSHOT_CONDITIONS: frozenset[str] = frozenset({"trail", "combo"})
_SNAPSHOT_GENERATIONS: frozenset[int] = frozenset({0, 50, N_GENERATIONS - 1})
_SNAPSHOT_REPLICATE: int = 0


@dataclass
class GenerationRecord:
    mean_n: float
    mean_d: float
    std_n: float
    std_d: float
    mean_s: float


@dataclass
class PopulationSnapshot:
    generation: int
    n: np.ndarray
    d: np.ndarray


@dataclass
class ReplicateResult:
    condition: str
    cell_line: str
    replicate: int
    trajectory: list[GenerationRecord] = field(default_factory=list)
    snapshots: list[PopulationSnapshot] = field(default_factory=list)
    resistance_generation: int | float = float("inf")


class Simulation:

    def __init__(self, seed: int | None = None) -> None:
        self._base_seed = seed
        self._evaluator = FitnessEvaluator()

    def _make_rng(self, condition: str, cell_line: str, replicate: int, seed_offset: int = 0) -> np.random.Generator:
        if self._base_seed is None:
            return np.random.default_rng()
        idx = (
            CONDITIONS.index(condition) * len(CELL_LINES) * N_REPLICATES
            + list(CELL_LINES).index(cell_line) * N_REPLICATES
            + replicate
        )
        return np.random.default_rng(self._base_seed + idx + seed_offset)

    def run_replicate(
        self,
        condition: str,
        cell_line: str,
        replicate: int,
        mutation_std: float = MUTATION_STD,
        seed_offset: int = 0,
    ) -> ReplicateResult:
        rng = self._make_rng(condition, cell_line, replicate, seed_offset)
        params = CELL_LINES[cell_line]
        pop = Population.initialise(
            POPULATION_SIZE, params["n_mean"], params["d_mean"], rng=rng
        )

        capture_snapshots = (
            condition in _SNAPSHOT_CONDITIONS and replicate == _SNAPSHOT_REPLICATE
        )
        result = ReplicateResult(condition=condition, cell_line=cell_line, replicate=replicate)

        for gen in range(N_GENERATIONS):
            if capture_snapshots and gen in _SNAPSHOT_GENERATIONS:
                n_snap, d_snap = pop.snapshot()
                result.snapshots.append(PopulationSnapshot(gen, n_snap, d_snap))

            survival = self._evaluator.evaluate(condition, pop)
            mean_s = float(survival.mean())

            stats = pop.stats()
            result.trajectory.append(GenerationRecord(
                mean_n=stats["mean_n"],
                mean_d=stats["mean_d"],
                std_n=stats["std_n"],
                std_d=stats["std_d"],
                mean_s=mean_s,
            ))

            if mean_s >= RESISTANCE_THRESHOLD and result.resistance_generation == float("inf"):
                result.resistance_generation = gen

            pop.select_and_reproduce(survival, mutation_std)

        return result

    def run_mutation_sweep(self) -> dict[float, list[ReplicateResult]]:
        """Run TRAIL and combo across mutation rates [0.01, 0.02, 0.05] for HPV- only."""
        sweep_results: dict[float, list[ReplicateResult]] = {}
        total = len(MUTATION_SWEEP_RATES) * len(SWEEP_CONDITIONS) * N_REPLICATES
        done = 0

        for rate_idx, mut_std in enumerate(MUTATION_SWEEP_RATES):
            seed_offset = 10000 * (rate_idx + 1)
            reps: list[ReplicateResult] = []
            for condition in SWEEP_CONDITIONS:
                for rep in range(N_REPLICATES):
                    reps.append(self.run_replicate(
                        condition, "hpv_neg", rep,
                        mutation_std=mut_std,
                        seed_offset=seed_offset,
                    ))
                    done += 1
                    print(f"\r  Mutation sweep: {done}/{total}", end="", flush=True)
            sweep_results[mut_std] = reps

        print()
        return sweep_results

    def run_all(self) -> list[ReplicateResult]:
        results: list[ReplicateResult] = []
        total = len(CONDITIONS) * len(CELL_LINES) * N_REPLICATES
        done = 0

        for cell_line in CELL_LINES:
            for condition in CONDITIONS:
                for rep in range(N_REPLICATES):
                    results.append(self.run_replicate(condition, cell_line, rep))
                    done += 1
                    print(f"\r  {done}/{total} runs complete", end="", flush=True)

        print()
        return results
