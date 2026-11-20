"""Unit tests for fitness functions against paper-derived validation targets."""

import numpy as np
import pytest

from research.simulation.fitness import FitnessEvaluator
from research.simulation.phenotype import Population

HPV_NEG_N = 0.5
HPV_NEG_D = 0.4


@pytest.fixture
def baseline() -> Population:
    return Population(
        n=np.array([HPV_NEG_N], dtype=np.float32),
        d=np.array([HPV_NEG_D], dtype=np.float32),
    )


@pytest.fixture
def evaluator() -> FitnessEvaluator:
    return FitnessEvaluator()


def test_control_near_full_survival(baseline, evaluator):
    s = evaluator.control(baseline)[0]
    assert s > 0.95, f"Control S={s:.3f} — expected > 0.95"


def test_trail_minimal_effect(baseline, evaluator):
    s = evaluator.trail(baseline)[0]
    assert 0.78 <= s <= 0.93, f"TRAIL S={s:.3f} — expected minimal killing at working dose"


def test_bap15_significant_killing(baseline, evaluator):
    s = evaluator.bap15(baseline)[0]
    assert s < 0.50, f"b-AP15 S={s:.3f} — expected significant reduction"


def test_combo_strong_killing(baseline, evaluator):
    s = evaluator.combo(baseline)[0]
    assert s < 0.25, f"Combo S={s:.3f} — expected strong synergistic killing"


def test_triple_at_least_as_lethal_as_combo(baseline, evaluator):
    assert evaluator.triple(baseline)[0] <= evaluator.combo(baseline)[0]


def test_combo_more_lethal_than_either_monotherapy(baseline, evaluator):
    s_trail = evaluator.trail(baseline)[0]
    s_bap15 = evaluator.bap15(baseline)[0]
    s_combo = evaluator.combo(baseline)[0]
    assert s_combo < s_trail, "Combo should kill more than TRAIL alone"
    assert s_combo < s_bap15, "Combo should kill more than b-AP15 alone"


def test_high_n_low_d_resists_trail(evaluator):
    pop = Population(
        n=np.array([0.9], dtype=np.float32),
        d=np.array([0.1], dtype=np.float32),
    )
    s = evaluator.trail(pop)[0]
    assert s > 0.85, f"High-XIAP cell should resist TRAIL, got S={s:.3f}"


def test_low_n_high_d_sensitive_to_trail(evaluator):
    pop = Population(
        n=np.array([0.1], dtype=np.float32),
        d=np.array([0.9], dtype=np.float32),
    )
    s = evaluator.trail(pop)[0]
    assert s < 0.30, f"Low-XIAP/high-DR5 cell should be TRAIL-sensitive, got S={s:.3f}"


def test_survival_ordering(baseline, evaluator):
    s = {c: evaluator.evaluate(c, baseline)[0] for c in ("control", "trail", "bap15", "combo", "triple")}
    assert s["control"] > s["trail"] > s["bap15"] > s["triple"]
    assert s["combo"] < s["bap15"]


def test_evaluate_raises_on_unknown_condition(baseline, evaluator):
    with pytest.raises(ValueError, match="Unknown condition"):
        evaluator.evaluate("unknown", baseline)


def test_feedback_raises_trail_survival(evaluator):
    # High DR5 activation → strong feedback → n_eff > n → partial XIAP rescue
    # Without feedback: A = d*(1-n) / (0.1+n) = 0.56/0.40 = 1.4 → S ≈ 0.009
    # With feedback: n_eff > n reduces DR5_expressed and raises XIAP, so S > 0.009
    pop = Population(
        n=np.array([0.3], dtype=np.float32),
        d=np.array([0.8], dtype=np.float32),
    )
    s_with_feedback = evaluator.trail(pop)[0]
    s_no_feedback = 1.0 / (1.0 + np.exp(6.0 * (0.8 * 0.7 / (0.1 + 0.3) - 0.62)))
    assert s_with_feedback > s_no_feedback, (
        f"Feedback should raise TRAIL survival: got {s_with_feedback:.4f} vs no-feedback {s_no_feedback:.4f}"
    )


def test_combo_ceiling_unchanged_at_full_adaptation(evaluator):
    # At n=1, d=0 (fully-adapted phenotype): d=0 eliminates DR5 contribution.
    # Pharmacological ceiling from b-AP15 alone: A = 0.75 → S ≈ 0.31.
    pop = Population(
        n=np.array([1.0], dtype=np.float32),
        d=np.array([0.0], dtype=np.float32),
    )
    s = evaluator.combo(pop)[0]
    assert 0.25 <= s <= 0.38, f"Ceiling at n=1, d=0 should be ≈0.31, got {s:.3f}"


def test_plasticity_not_heritable(evaluator):
    # Fitness evaluation must not modify the population's stored genome values.
    pop = Population(
        n=np.array([0.3], dtype=np.float32),
        d=np.array([0.8], dtype=np.float32),
    )
    n_before = pop.n[0]
    _ = evaluator.trail(pop)
    assert pop.n[0] == n_before, (
        f"Genomic n must be unchanged after fitness evaluation: {pop.n[0]} != {n_before}"
    )
