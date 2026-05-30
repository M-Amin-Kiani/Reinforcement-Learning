# Reinforcement Learning Assignment --- From MDP to Dynamic Programming

A complete educational Reinforcement Learning assignment repository
built around a connected example inspired by Sutton & Barto. The goal of
this project is not only to solve the assignment, but also to teach
every concept clearly and intuitively.

------------------------------------------------------------------------

## Project Goal

This repository explains Reinforcement Learning concepts step-by-step
using one consistent example across all assignment sections.

Instead of only showing formulas, this project answers:

-   What does this equation mean?
-   Why is it correct?
-   Where does it come from?
-   How do we compute it step-by-step?

This repository is designed for students who want to **understand
Reinforcement Learning deeply**, not only pass the assignment.

------------------------------------------------------------------------

## Topics Covered

### 1. Finite Markov Decision Process (MDP)

We begin with a custom version of Sutton & Barto Example 3.3 (Recycling
Robot).

Concepts covered:

-   States
-   Actions
-   Transition probabilities
-   Reward distributions
-   Non-deterministic environments
-   Non-uniform probability distributions
-   Markov Property

Core idea:

> The future depends only on the current state and action --- not the
> full history.

------------------------------------------------------------------------

### 2. Markov Reward Process (MRP)

After fixing a policy, the MDP becomes an MRP.

You will learn:

-   Policy-induced transition matrices
-   Expected rewards
-   State-value functions
-   Bellman expectation intuition

Core equations:

State transition under policy:

pπ(s'\|s) = Σ_a π(a\|s)p(s'\|s,a)

Expected reward:

rπ(s) = Σ_a π(a\|s) Σ\_(s',r) r p(s',r\|s,a)

------------------------------------------------------------------------

### 3. Monte Carlo Evaluation

We estimate state values using sampled episodes.

Methods:

#### First-Visit Monte Carlo

Uses only the first visit of a state inside an episode.

#### Every-Visit Monte Carlo

Uses every occurrence of a state.

Concepts explained:

-   Return
-   Sampling
-   Estimation
-   Convergence intuition

Key idea:

> Repeated averaging approaches the expected value.

------------------------------------------------------------------------

### 4. Bellman Equation & Policy Iteration

We move from sampled estimation toward recursive exact computation.

Topics:

-   Bellman expectation equation
-   Policy evaluation
-   Policy improvement theorem
-   State-action value function

Definition:

Q(s,a) = immediate reward + discounted future value

Policy improvement theorem:

If:

qπ(s,π'(s)) ≥ vπ(s)

Then:

vπ'(s) ≥ vπ(s)

Meaning:

> Choosing locally better actions cannot make the overall policy worse.

------------------------------------------------------------------------

### 5. Dynamic Programming (Finite Horizon)

We solve a finite-horizon planning problem using backward dynamic
programming.

Topics:

-   Horizon-dependent value functions
-   Bellman recursion
-   Backward optimization

Core recursion:

V_t(s)=max(reward_now + V\_(t−1)(next_state))

Boundary condition:

V_0(s)=0

Meaning:

> If no time remains, no future reward can be collected.

------------------------------------------------------------------------

## Learning Flow

``` text
MDP
 ↓
Policy
 ↓
MRP
 ↓
Monte Carlo Evaluation
 ↓
Bellman Equation
 ↓
Policy Evaluation
 ↓
Policy Improvement
 ↓
Dynamic Programming
```

------------------------------------------------------------------------

## Repository Structure

``` text
.
├── report/
│   └── assignment_report.pdf
├── notebooks/
│   └── reinforcement_learning_assignment.ipynb
├── assets/
├── README.md
```

------------------------------------------------------------------------

## Why This Repository Is Different

This project emphasizes:

-   Intuition before equations
-   Step-by-step derivations
-   Mathematical correctness
-   Educational explanations
-   End-to-end consistency across all sections

Instead of memorizing formulas, you will understand **why they work**.

------------------------------------------------------------------------

## References

-   Sutton & Barto --- Reinforcement Learning: An Introduction
-   University course materials
-   Reinforcement Learning assignment implementation

------------------------------------------------------------------------

## Educational Philosophy

If you have ever asked:

> "Why does this equation work?"

this repository was written for you.

------------------------------------------------------------------------

## License

MIT License
