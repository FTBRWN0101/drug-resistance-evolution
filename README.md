# Evolutionary dynamics of drug resistance on a signalling fitness landscape

An evolutionary simulation of head and neck cancer cells under five drug treatments.
It tests whether combination therapy prevents resistance by reshaping the fitness landscape, rather than by killing more cells.
Each cell carries two heritable traits, NFκB drive and DR5 sensitivity, and its survival under each treatment comes from fitness functions calibrated to Brown et al. (2026).

Built for an MSc module at the University of Sussex (2026).

## What it found

Under TRAIL alone, resistance (mean survival of at least 0.85) emerged in all 20 replicates, at a median of generation 3.
Under b-AP15, b-AP15 + TRAIL and the triple combination, no replicate reached resistance within 400 generations, for either HPV status and at all three mutation rates (one-sided Mann-Whitney U test, TRAIL against b-AP15 + TRAIL: p < 0.001).
The population still adapts under combination: both traits sweep to the high-NFκB, zero-DR5 corner within about 150 generations, where survival plateaus near 0.22.
No phenotype can beat the analytical ceiling of 0.31, far below the resistance threshold.
Combination therapy prevents resistance by removing every reachable high-fitness phenotype, not by killing more cells.

## Run it

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync
uv run python run.py
```

This runs 200 simulations (5 treatments, 2 cell lines, 20 replicates each) and a 120-run mutation-rate sweep, then writes eight figures to `figures/`.
It takes about a minute.
Every run is seeded (`seed=42` in `run.py`), but exact numbers can vary across machines and library versions.

## How the code is laid out

- `research/simulation/phenotype.py`: the cell genome, the population, and selection with mutation.
- `research/simulation/fitness.py`: survival under each treatment.
- `research/simulation/drug_resistance_experiment.py`: the generational loop, the replicates and the mutation-rate sweep.
- `research/simulation/plots.py`: the figures and the Mann-Whitney U test.
- [METHODS.md](METHODS.md): the model's equations, its parameters and their sources, and the calibration targets.

## Test it

```bash
uv run pytest
```

The tests check the fitness functions against the calibration targets in [METHODS.md](METHODS.md), such as survival near 0.85 for HPV-negative cells under TRAIL.

Tested on Linux and Windows with Python 3.12.

## Write-up

The full write-up isn't published here.
If you'd like to read it, email me at ftbrwn01@gmail.com.

## Licence

The code is under the Apache License 2.0; see [LICENSE](LICENSE).
Copyright 2026 Findlay Brown.

## Citation

To cite this code, use **Cite this repository** on the GitHub page, which reads [CITATION.cff](CITATION.cff).
The model's parameters come from:

Brown, F. T. A. et al. (2026). b-AP15 enhances TRAIL-induced cell death in HNSCC via the induction of ROS/JNK/DR5 signalling. *Cancer Gene Therapy* 33(5): 533-546. https://doi.org/10.1038/s41417-026-01038-3
