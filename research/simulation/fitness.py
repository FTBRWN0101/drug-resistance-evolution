"""Fitness functions mapping phenotype to survival probability per treatment."""

from __future__ import annotations

import numpy as np

from .phenotype import (
    DR5_AMPLIFICATION,
    DR5_FEEDBACK_STRENGTH,
    IAP_ANTAGONIST_EFFICACY,
    Population,
    SOD2_REDUCTION,
    TAK1_FRACTION,
    XIAP_REDUCTION,
)

# Shifted sigmoid parameters: S(A) = 1 / (1 + exp(K * (A - A50)))
# A50 = 0.62 places the HPV- TRAIL baseline (A≈0.33) at S≈0.85
K: float = 6.0
A50: float = 0.62


def _sigmoid(a: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(K * (a - A50)))).astype(np.float32)


def _bap15_expression(
    pop: Population,
    xiap_mod: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (n_eff, xiap_eff, sod2_eff, dr5_expressed) under b-AP15 conditions.

    Applies DR5→NFκB feedback then splits XIAP→JNK suppression across the
    TAK1-ubiquitination and SOD2/ROS arms, each reduced at their own drug rate.
    xiap_mod: XIAP_REDUCTION for combo; XIAP_REDUCTION * IAP_ANTAGONIST_EFFICACY for triple.
    """
    n, d = pop.n, pop.d
    dr5_basal = d * (1.0 - n)
    n_eff = np.clip(n + DR5_FEEDBACK_STRENGTH * dr5_basal, 0.0, 1.0)
    xiap_eff = n_eff * xiap_mod
    sod2_eff = n_eff * SOD2_REDUCTION
    suppression = TAK1_FRACTION * xiap_eff + (1.0 - TAK1_FRACTION) * sod2_eff
    dr5_expressed = d * (1.0 - suppression)
    return n_eff, xiap_eff, sod2_eff, dr5_expressed


class FitnessEvaluator:
    """Computes per-cell survival probability for each treatment condition."""

    def control(self, pop: Population) -> np.ndarray:
        return _sigmoid(np.zeros(len(pop), dtype=np.float32))

    def trail(self, pop: Population) -> np.ndarray:
        n, d = pop.n, pop.d
        dr5_basal = d * (1.0 - n)
        n_eff = np.clip(n + DR5_FEEDBACK_STRENGTH * dr5_basal, 0.0, 1.0)
        dr5_expressed = d * (1.0 - n_eff)
        return _sigmoid(dr5_expressed / (0.1 + n_eff))

    def bap15(self, pop: Population) -> np.ndarray:
        xiap, sod2, _ = pop.basal_expression()
        xiap_eff = xiap * XIAP_REDUCTION
        sod2_eff = sod2 * SOD2_REDUCTION
        a = 0.5 * (1.0 - sod2_eff) + 0.5 * (1.0 - xiap_eff)
        return _sigmoid(a)

    def combo(self, pop: Population) -> np.ndarray:
        _, xiap_eff, sod2_eff, dr5_expressed = _bap15_expression(pop, XIAP_REDUCTION)
        amplification = 1.0 + (DR5_AMPLIFICATION - 1.0) * (1.0 - sod2_eff)
        dr5_eff = dr5_expressed * amplification
        a = 0.5 * (1.0 - sod2_eff) + 0.5 * (1.0 - xiap_eff) + dr5_eff / (0.1 + xiap_eff)
        return _sigmoid(a)

    def triple(self, pop: Population) -> np.ndarray:
        _, xiap_eff, sod2_eff, dr5_expressed = _bap15_expression(
            pop, XIAP_REDUCTION * IAP_ANTAGONIST_EFFICACY
        )
        amplification = 1.0 + (DR5_AMPLIFICATION - 1.0) * (1.0 - sod2_eff)
        dr5_eff = dr5_expressed * amplification
        a = 0.5 * (1.0 - sod2_eff) + 0.5 * (1.0 - xiap_eff) + dr5_eff / (0.1 + xiap_eff)
        return _sigmoid(a)

    def evaluate(self, condition: str, pop: Population) -> np.ndarray:
        dispatch = {
            "control": self.control,
            "trail": self.trail,
            "bap15": self.bap15,
            "combo": self.combo,
            "triple": self.triple,
        }
        if condition not in dispatch:
            raise ValueError(f"Unknown condition: {condition!r}")
        return dispatch[condition](pop)
