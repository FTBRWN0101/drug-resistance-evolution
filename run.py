"""Entry point: run the full drug resistance experiment and generate all figures."""

from research.simulation.drug_resistance_experiment import Simulation
from research.simulation.plots import generate_all, generate_mutation_figures

if __name__ == "__main__":
    sim = Simulation(seed=42)

    print("Running 200 simulations (5 conditions × 2 cell lines × 20 replicates)...")
    results = sim.run_all()
    generate_all(results)

    print("Running mutation rate sweep (2 conditions × 3 rates × 20 replicates)...")
    sweep_results = sim.run_mutation_sweep()
    generate_mutation_figures(sweep_results)
