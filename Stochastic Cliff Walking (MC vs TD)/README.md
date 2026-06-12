# Reinforcement Learning: Stochastic Cliff Walking (MC vs TD)

This repository contains a **Google Colab-ready** project for building, evaluating, and comparing fundamental **Reinforcement Learning (RL)** algorithms on a custom **Stochastic Cliff Walking** environment.

The project systematically covers both **Prediction (Evaluation)** and **Control** tasks using methods from *Reinforcement Learning: An Introduction* by Sutton & Barto (2nd Edition).

## What This Project Includes

- **Monte Carlo (MC)** methods
  - First-visit MC
  - Every-visit MC
  - Off-policy evaluation with Importance Sampling
- **Temporal Difference (TD)** methods
  - TD(0)
  - n-step TD
- **TD Control algorithms**
  - SARSA (On-policy)
  - Q-Learning (Off-policy)
  - Expected SARSA
  - Double Q-Learning

The project compares these methods in terms of:

- convergence rate
- variance and bias
- stability during learning
- final learned policy safety
- susceptibility to maximization bias

---

## Problem Setting

The environment is a modified 4×12 grid-world based on the classic Cliff Walking problem.

### Grid Layout

- **Start (S):** Bottom-left corner
- **Goal (G):** Bottom-right corner
- **Cliff (C):** All cells between Start and Goal on the bottom row
- **Free cells (F):** All other cells

### Stochasticity

A slip mechanism is added to make the environment more realistic:

- When the agent selects an action, it moves in the intended direction with probability `1 - slip_rate`.
- With probability `slip_rate`, it slips into one of the other directions.
- In this project, `slip_rate = 0.1` is used in the main experiments.

### Reward Structure

- **-1** for each normal step
- **-100** for falling into the cliff  
  The agent is sent back to the start state, but the episode does **not** terminate.
- **0** for reaching the goal  
  The episode terminates immediately.

### Why This Environment Is Interesting

This setting creates a classic **risk vs. reward** trade-off:

- The path along the cliff edge is short but dangerous.
- The upper route is longer but much safer under stochastic transitions.

---

## Key Features

- Complete pipeline for **environment setup, training, evaluation, and visualization**
- **Off-policy evaluation** using:
  - Ordinary Importance Sampling (OIS)
  - Weighted Importance Sampling (WIS)
- **Bias–variance analysis** using n-step TD
- **Ground truth state values** computed with iterative policy evaluation
- **Policy visualization** for learned control strategies
- **Learning curves** with confidence bands across multiple independent runs
- Empirical comparison of **Q-Learning overestimation** and **Double Q-Learning mitigation**

---

## Repository Structure

```text
.
├── notebooks/
│   └── Stochastic_Cliff_Walking_RL.ipynb
├── env/
│   └── StochasticCliffWalking.py
├── results/
│   ├── plots/
│   └── policy_grids/
├── README.md
└── LICENSE
```

---

## Environment File

The custom environment is implemented in:

```text
StochasticCliffWalking.py
```

This file defines a Gym-style environment for the stochastic cliff walking task and can be used directly in Google Colab.

---

## Quick Start (Google Colab)

### 1) Open the notebook
Open the main notebook in Google Colab:

```text
notebooks/Stochastic_Cliff_Walking_RL.ipynb
```

### 2) Install dependencies
Run the following command in Colab:

```bash
pip install numpy matplotlib gym
```

If you are using Gymnasium instead of Gym, adapt the imports accordingly.

### 3) Run the notebook cells sequentially
The notebook is organized into three main parts:

- **Part 1: Monte Carlo**
  - episode generation
  - epsilon-soft control
  - importance sampling evaluation

- **Part 2: TD Prediction**
  - TD(0) vs MC
  - n-step TD
  - ground truth state values via dynamic programming

- **Part 3: TD Control**
  - SARSA
  - Q-Learning
  - Expected SARSA
  - Double Q-Learning

All plots and visualizations are generated inline.

---

## Algorithms and Intuition

### Monte Carlo vs. TD(0)

**Every-visit MC**
- Updates state values using the full return from complete episodes.
- Pros: unbiased target in episodic settings.
- Cons: high variance and slower learning.

**TD(0)**
- Updates using the immediate reward plus the bootstrap estimate of the next state.
- Pros: lower variance and faster learning.
- Cons: introduces bias through bootstrapping.

### SARSA vs. Q-Learning

**SARSA**
- On-policy method
- Learns the value of the policy being executed, including exploration
- In risky environments, it often learns a safer route away from the cliff

**Q-Learning**
- Off-policy method
- Learns the greedy target policy while behaving with exploration
- Often prefers the shortest route near the cliff
- Can suffer from instability and overestimation in stochastic settings

### Expected SARSA

Expected SARSA replaces sampling of the next action with the expected value under the current policy.

Advantages:

- lower variance than standard SARSA
- smoother learning curves
- often more stable with larger learning rates

### Double Q-Learning

Double Q-Learning uses two separate Q-tables to reduce maximization bias.

Main idea:

- one table selects the action
- the other evaluates it

This decoupling helps reduce the positive bias caused by the `max` operator in standard Q-Learning.

---

## Evaluation Metrics

The repository includes the following evaluation outputs:

- **Average Return** over episodes
- **Confidence bands** over multiple runs
- **RMSE vs. Ground Truth** for prediction methods
- **OIS vs. WIS convergence plots**
- **Learned policy grids**
- **Maximization bias curves** for Q-Learning vs. Double Q-Learning

---

## Reproducibility Checklist

- Fixed hyperparameters across control algorithms
  - `alpha = 0.1`
  - `epsilon = 0.1`
  - `gamma = 0.99`
- Ground truth values computed using **iterative policy evaluation**
- Results averaged over **10 independent runs** where required
- Incremental updates used for importance sampling and value estimation to reduce memory usage

---

## Reference

This project is based on concepts from:

**Sutton, R. S., & Barto, A. G. (2018).**  
*Reinforcement Learning: An Introduction (2nd Edition).*  
MIT Press.

---

## License

Add the appropriate license file for your repository if you plan to publish or share the project publicly.

---

## Author

**Amin Kiani**
