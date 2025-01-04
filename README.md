# Resonance Reduction Simulation in Dynamic Networks

## Project Description

This simulation models the behavior of dynamic networks under adversarial resonance attacks and evaluates methods to mitigate their effects through **eigenspectrum optimization**. Using C++, the simulation focuses on the network's response to external forcing, assessing how resonance can be controlled by adjusting network properties through gradient descent optimization.

## Key Objectives

1. Simulate network dynamics under second-order differential equations with adversarial forcing.
2. Evaluate network vulnerability before and after applying gradient descent optimization to network edge weights.
3. Generate visualizations and performance metrics for the network's dynamic behavior.

## Code Overview

The simulation is implemented in `gdMain.cpp` and is part of a larger system with custom modules for graph dynamics, force application, and plotting.

### Features
1. **Graph Construction**:
   - Initializes the network as a graph with specified edge weights.
2. **Dynamic Simulation**:
   - Simulates second-order dynamics with user-defined parameters like damping, stiffness, and external forcing.
3. **Gradient Descent Optimization**:
   - Optimizes edge weights to flatten the eigenvalue spectrum, reducing the network's susceptibility to resonance.
4. **Visualization**:
   - Generates plots for network states before and after optimization.

### Simulation Parameters
- **Number of Nodes (`numNodes`)**: Size of the network.
- **Damping (`damping`)**: Factor to control signal dissipation.
- **Stiffness (`stiffness`)**: Parameter for edge weight adjustments.
- **H Parameter (`h`)**: Spread of forcing frequency distribution.
- **Simulation Time (`simulationTime`)**: Duration of each simulation.
