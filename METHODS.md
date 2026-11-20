# Methods

This file sets out the model the code implements: its equations, its parameter values and their sources, and the calibration targets the tests check.
Equation numbers follow the original write-up.
Names in backticks are the matching identifiers in the code.

## Cell phenotype

Each cell carries two heritable traits, both in [0, 1]:

- **NFκB drive** (`n`): the activity of the NFκB pathway, which drives expression of the anti-apoptotic proteins XIAP and SOD2.
- **DR5 sensitivity** (`d`): the responsiveness of the ROS→JNK→DR5 induction axis.

Fitness evaluation derives three protein levels from the traits (`Population.basal_expression`):

```math
\mathrm{XIAP}_{\mathrm{basal}} = n \qquad (1)
```

```math
\mathrm{SOD2}_{\mathrm{basal}} = n \qquad (2)
```

```math
\mathrm{DR5}_{\mathrm{basal}} = d\,(1 - n) \qquad (3)
```

The (1 - n) term in Eq. 3 reflects NFκB suppressing JNK through XIAP-dependent TAK1 degradation, and DR5 induction needs JNK.
So a cell cannot maximise DR5 sensitivity and XIAP protection independently.

## DR5 to NFκB feedback

When TRAIL engages DR5, the signalling raises effective NFκB activity above the cell's genomic set-point:

```math
n_{\mathrm{eff}} = \mathrm{clip}\left(n + \beta\,\mathrm{DR5}_{\mathrm{basal}},\ 0,\ 1\right), \qquad \beta = 0.15 \qquad (4)
```

This plasticity acts within a cell's lifetime only: offspring inherit `n`, not `n_eff`.
It applies in the three conditions with TRAIL (TRAIL, b-AP15 + TRAIL and the triple combination).
The value of β (`DR5_FEEDBACK_STRENGTH`) keeps the HPV-negative TRAIL baseline at S ≈ 0.85.

## Drug effects

b-AP15 inhibits USP14, which lowers NFκB activity, depletes XIAP and SOD2, and releases ROS/JNK signalling to amplify DR5.
The model applies these effects as multipliers during fitness evaluation (constants in `phenotype.py`):

| Modifier | Value | Constant | Source in Brown et al. (2026) |
| --- | --- | --- | --- |
| XIAP reduction (b-AP15) | ×0.30 | `XIAP_REDUCTION` | Fig. 6B: about 70% reduction at 500 nM |
| SOD2 reduction (b-AP15) | ×0.20 | `SOD2_REDUCTION` | Fig. 6B: about 80% reduction at 500 nM |
| DR5 amplification (b-AP15) | ×3.5 | `DR5_AMPLIFICATION` | Fig. 2C: about 3-4× induction at 500 nM |
| IAP antagonist efficacy | ×0.15 | `IAP_ANTAGONIST_EFFICACY` | Fig. 8A: over 90% cIAP1 reduction |
| TAK1 share of JNK suppression | 0.5 | `TAK1_FRACTION` | Modelling assumption, not measured |

## Survival

Survival probability S is a shifted sigmoid of the net apoptotic signal A (`K`, `A50` and `_sigmoid` in `fitness.py`):

```math
S(A) = \frac{1}{1 + e^{k\,(A - A_{50})}}, \qquad k = 6,\quad A_{50} = 0.62 \qquad (5)
```

A50 is set so that the HPV-negative baseline phenotype (n = 0.5, d = 0.4) under TRAIL gives S ≈ 0.85.
That matches the minimal killing at working doses in Fig. 1A of Brown et al. (2026).
With the feedback of Eq. 4 active, the same phenotype gives S ≈ 0.87, still within the target.

Each treatment computes A differently (methods of `FitnessEvaluator`).

**No treatment** (`control`): A = 0 for every cell, so S ≈ 0.98 and there is no selective gradient.

**TRAIL** (`trail`): TRAIL binds DR5 to start apoptosis, and XIAP blocks caspase activation downstream.

```math
\mathrm{DR5}_{\mathrm{expressed}} = d\,(1 - n_{\mathrm{eff}}) \qquad (6)
```

```math
A_{\mathrm{TRAIL}} = \frac{\mathrm{DR5}_{\mathrm{expressed}}}{0.1 + n_{\mathrm{eff}}} \qquad (7)
```

**b-AP15** (`bap15`): no TRAIL is present, so the feedback does not fire and the basal levels apply.

```math
A_{\mathrm{bAP15}} = 0.5\,(1 - \mathrm{SOD2}_{\mathrm{eff}}) + 0.5\,(1 - \mathrm{XIAP}_{\mathrm{eff}}) \qquad (8)
```

Here SOD2_eff = SOD2_basal × 0.20 and XIAP_eff = XIAP_basal × 0.30.

**b-AP15 + TRAIL** (`combo`, using `_bap15_expression`): the b-AP15 modifiers act on the feedback-adjusted NFκB level.

```math
\mathrm{XIAP}_{\mathrm{eff}} = n_{\mathrm{eff}} \times 0.30 \qquad (9)
```

```math
\mathrm{SOD2}_{\mathrm{eff}} = n_{\mathrm{eff}} \times 0.20 \qquad (10)
```

NFκB suppresses JNK through two arms, direct TAK1 ubiquitination (XIAP-dependent) and SOD2/ROS buffering, weighted equally:

```math
\mathrm{suppr} = \tfrac{1}{2}\,\mathrm{XIAP}_{\mathrm{eff}} + \tfrac{1}{2}\,\mathrm{SOD2}_{\mathrm{eff}} \qquad (11)
```

```math
\mathrm{DR5}_{\mathrm{expressed}} = d\,(1 - \mathrm{suppr}) \qquad (12)
```

The ×3.5 amplification comes from ROS/JNK release as SOD2 falls, so it scales with SOD2 loss rather than acting as a flat multiplier:

```math
\alpha = 1 + (3.5 - 1)\,(1 - \mathrm{SOD2}_{\mathrm{eff}}) \qquad (13)
```

```math
\mathrm{DR5}_{\mathrm{eff}} = \alpha\,\mathrm{DR5}_{\mathrm{expressed}} \qquad (14)
```

```math
A_{\mathrm{combo}} = 0.5\,(1 - \mathrm{SOD2}_{\mathrm{eff}}) + 0.5\,(1 - \mathrm{XIAP}_{\mathrm{eff}}) + \frac{\mathrm{DR5}_{\mathrm{eff}}}{0.1 + \mathrm{XIAP}_{\mathrm{eff}}} \qquad (15)
```

Synergy follows because b-AP15 raises DR5_eff and lowers XIAP_eff at the same time.

**Triple combination, b-AP15 + TRAIL + IAP antagonist** (`triple`): the antagonist suppresses XIAP function on top of the b-AP15 reduction.
All other terms are as in the combination.

```math
\mathrm{XIAP}_{\mathrm{eff}} = n_{\mathrm{eff}} \times 0.30 \times 0.15 \qquad (16)
```

## Survival ceilings

At the fully adapted phenotype (n = 1, d = 0), the DR5 term vanishes and only the b-AP15 expression effects remain.
No other phenotype survives better, so these values cap what selection can reach:

```math
A_{\mathrm{bAP15}} = 0.5\,(1 - 0.20) + 0.5\,(1 - 0.30) = 0.75 \quad\Rightarrow\quad S \approx 0.31 \qquad (17)
```

```math
A_{\mathrm{triple}} = 0.5\,(1 - 0.20) + 0.5\,(1 - 0.045) = 0.8775 \quad\Rightarrow\quad S \approx 0.18 \qquad (18)
```

Eq. 17 is the ceiling for b-AP15 alone and for b-AP15 + TRAIL.
Both ceilings sit far below the resistance threshold of 0.85.

## Evolutionary algorithm

The simulation is a synchronous generational evolutionary algorithm with asexual reproduction and no crossover (`Population.select_and_reproduce`).
Each generation:

1. Compute every cell's survival S under the current treatment, and record the population's mean and standard deviation of each trait and its mean survival.
2. Each cell survives with probability S. If no cell survives, every cell counts as a survivor, which guards against extinction.
3. Sample N parents from the survivors, with replacement.
4. Each offspring takes its parent's traits plus independent Gaussian noise with standard deviation σm on each trait, clipped to [0, 1].

| Parameter | Value | Constant in `drug_resistance_experiment.py` |
| --- | --- | --- |
| Population size N | 500 | `POPULATION_SIZE` |
| Generations | 400 | `N_GENERATIONS` |
| Mutation standard deviation σm | 0.02 | `MUTATION_STD` |
| Replicates per condition | 20 | `N_REPLICATES` |
| Resistance threshold | 0.85 | `RESISTANCE_THRESHOLD` |

## Starting populations

Two cell lines reflect the baseline difference by HPV status (`CELL_LINES`).
Both traits start from a normal distribution around the mean below, with standard deviation 0.1, clipped to [0, 1].

| Cell line | Mean n | Mean d |
| --- | --- | --- |
| HPV-negative (UMSCC1, UMSCC22A) | 0.50 | 0.40 |
| HPV-positive (UMSCC47) | 0.55 | 0.30 |

## Experimental design

- **Main experiment:** 5 treatments × 2 cell lines × 20 replicates gives 200 runs of 400 generations each.
- **Mutation-rate sweep:** σm of 0.01, 0.02 and 0.05 for TRAIL and b-AP15 + TRAIL, HPV-negative cells only, gives 120 more runs.
- **Resistance:** the first generation in which mean population survival reaches 0.85. A run that never reaches it counts as non-resistant.
- **Statistics:** a one-sided Mann-Whitney U test of resistance times, with the alternative that resistance emerges earlier under TRAIL than under b-AP15 + TRAIL (in `plots.py`). Non-resistant runs enter the rank test at generation 415, after every possible resistance time.
- **Seeds:** `Simulation(seed=42)` gives each run its own seed: the base seed plus an offset for its condition, cell line and replicate. The sweep adds a further 10,000 per mutation rate.

## Calibration targets

The tests in `tests/test_fitness.py` check the fitness functions at single phenotypes.

| Check | Phenotype (n, d) | Target |
| --- | --- | --- |
| No treatment | (0.5, 0.4) | S > 0.95 |
| TRAIL | (0.5, 0.4) | 0.78 ≤ S ≤ 0.93 |
| b-AP15 | (0.5, 0.4) | S < 0.50 |
| b-AP15 + TRAIL | (0.5, 0.4) | S < 0.25, and below both single drugs |
| Triple combination | (0.5, 0.4) | S no higher than b-AP15 + TRAIL |
| Ordering | (0.5, 0.4) | No treatment > TRAIL > b-AP15 > triple, and b-AP15 + TRAIL < b-AP15 |
| TRAIL resistance | (0.9, 0.1) | S > 0.85 under TRAIL |
| TRAIL sensitivity | (0.1, 0.9) | S < 0.30 under TRAIL |
| Feedback | (0.3, 0.8) | S under TRAIL is higher than without the feedback |
| Combination ceiling | (1, 0) | 0.25 ≤ S ≤ 0.38 (Eq. 17) |
| Plasticity | (0.3, 0.8) | Fitness evaluation leaves the genome's n unchanged |
| Unknown treatment | Any | Raises `ValueError` |

## Source

Brown, F. T. A. et al. (2026). b-AP15 enhances TRAIL-induced cell death in HNSCC via the induction of ROS/JNK/DR5 signalling. *Cancer Gene Therapy* 33(5): 533-546. https://doi.org/10.1038/s41417-026-01038-3
