"""Cell phenotype model: genome parameters and population dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Drug effect constants — Brown et al. (2026), Fig. 6B and Fig. 2C
XIAP_REDUCTION = 0.30
SOD2_REDUCTION = 0.20
DR5_AMPLIFICATION = 3.5
IAP_ANTAGONIST_EFFICACY = 0.15

# Mechanistic split: fraction of XIAP→JNK suppression via direct TAK1 ubiquitination
# vs. the XIAP→SOD2→ROS→JNK arm. Equal split — modelling assumption, not measured.
TAK1_FRACTION = 0.5

# DR5→NFκB pro-survival feedback strength (Edge G, dataset_card_v2).
# Within-lifetime only: n_eff = n + DR5_FEEDBACK_STRENGTH * dr5_basal.
# Offspring inherit n, not n_eff. Value chosen to preserve TRAIL calibration at HPV- baseline.
DR5_FEEDBACK_STRENGTH = 0.15


@dataclass(frozen=True)
class Cell:
    """Single-cell genome for testing and validation."""

    n: float  # NFκB drive [0, 1]
    d: float  # DR5 pathway sensitivity [0, 1]

    def basal_expression(self) -> tuple[float, float, float]:
        """Return (xiap, sod2, dr5) at baseline."""
        return self.n, self.n, self.d * (1.0 - self.n)


class Population:
    """Vectorised population stored as numpy arrays for efficient evolution."""

    def __init__(
        self,
        n: np.ndarray,
        d: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n = np.asarray(n, dtype=np.float32)
        self.d = np.asarray(d, dtype=np.float32)
        self.rng = rng if rng is not None else np.random.default_rng()

    @classmethod
    def initialise(
        cls,
        size: int,
        n_mean: float,
        d_mean: float,
        init_std: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> Population:
        _rng = rng if rng is not None else np.random.default_rng()
        n = np.clip(_rng.normal(n_mean, init_std, size), 0.0, 1.0).astype(np.float32)
        d = np.clip(_rng.normal(d_mean, init_std, size), 0.0, 1.0).astype(np.float32)
        return cls(n, d, rng=_rng)

    def __len__(self) -> int:
        return len(self.n)

    def basal_expression(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return vectorised (xiap, sod2, dr5) arrays."""
        return self.n, self.n, self.d * (1.0 - self.n)

    def select_and_reproduce(self, survival: np.ndarray, mutation_std: float) -> None:
        """Stochastic selection then asexual reproduction back to original size."""
        size = len(self)
        alive = self.rng.random(size, dtype=np.float32) < survival
        survivor_idx = np.where(alive)[0]

        if len(survivor_idx) == 0:
            survivor_idx = np.arange(size)

        idx = self.rng.choice(survivor_idx, size=size, replace=True)
        noise = self.rng.normal(0.0, mutation_std, (2, size)).astype(np.float32)
        self.n = np.clip(self.n[idx] + noise[0], 0.0, 1.0)
        self.d = np.clip(self.d[idx] + noise[1], 0.0, 1.0)

    def stats(self) -> dict[str, float]:
        return {
            "mean_n": float(self.n.mean()),
            "mean_d": float(self.d.mean()),
            "std_n": float(self.n.std()),
            "std_d": float(self.d.std()),
        }

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        return self.n.copy(), self.d.copy()
