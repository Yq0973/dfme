# Role and Context
You are an expert AI algorithm engineer, specializing in Power System Security, Adversarial Machine Learning, and Mathematical Optimization. Your task is to build a highly modular Python codebase to validate a "Physical-Topology Dual-Driven Black-Box Adversarial Attack" framework against a DC power system State Estimation (SE) anomaly detector.

# Project Objective (Phase 1)
Build an end-to-end simulation pipeline using a pure Python stack. We will test the attack framework on the IEEE 14-bus system. The target model is a simple MLP-based Neural Attack Detector (NAD). The attack algorithm will be P-ZOO (Projected Zeroth-Order Optimization).

# Tech Stack Requirements
- Power System Environment: `pandapower` or `pypower` (to extract topology, Jacobian matrix, and generate load data).
- Deep Learning: `PyTorch` (for the MLP NAD model).
- Optimization & Math: `numpy`, `scipy.optimize` (or `cvxopt` for Quadratic Programming projection), `networkx` (for graph Laplacian).

# Mathematical Formulation
The optimization variable is the state perturbation vector $c \in \mathbb{R}^n$ (node voltage angles).
The attack vector injected into measurements is $a = Hc \in \mathbb{R}^m$.
The final manipulated measurement is $z' = z + a = z + Hc$.

**1. Objective Function (to maximize):**
$$J(c) = \mathcal{N}(z + Hc) - \lambda \cdot c^T L c$$
Where:
- $\mathcal{N}(\cdot)$ is the black-box MLP model outputting the probability [0, 1] that the input is a "Normal" state.
- $L$ is the unnormalized Graph Laplacian matrix of the grid ($n \times n$).
- $\lambda$ is a hyperparameter (e.g., 0.1).

**2. Physical Constraints (Convex Polytope):**
The line flow must not exceed the thermal limits $F_{max}$.
$$-F_{max} \le M(x + c) \le F_{max}$$
Where:
- $M \in \mathbb{R}^{l \times n}$ is the branch shift factor matrix (PTDF matrix mapping angles to line flows).
- $x \in \mathbb{R}^n$ is the base case voltage angle vector.
This can be rewritten as standard linear inequalities $A_{ineq} c \le b_{ineq}$:
$A_{ineq} = [M; -M]$ (vertical concatenation)
$b_{ineq} = [F_{max} - Mx; F_{max} + Mx]$

# Module Breakdown & Implementation Steps

Please implement the framework step-by-step, dividing the code into the following specific modules. **Show me the code for each module sequentially, waiting for my approval before moving to the next.**

## Module 1: Power System Environment & Data Generator (`env.py`)
- Load the IEEE 14-bus system using `pandapower.networks.case14()`.
- Run DC power flow to get the base state $x$ (angles) and base measurements $z$ (active power injections and line flows).
- Extract/Calculate the constant matrices:
  - $H$ (Measurement Jacobian mapping $x$ to $z$).
  - $M$ (Branch flow matrix mapping $x$ to line flows).
  - $F_{max}$ (Line thermal limits. If not defined in case14, set a reasonable threshold, e.g., 120% of base case flows).
  - $L$ (Laplacian matrix $D - A$, where $A$ is the adjacency matrix weighted by branch susceptance).
- Create a data generator function that adds Gaussian noise (e.g., 30dB SNR) to $z$ to simulate 1000 normal samples, and generates 1000 basic FDIA samples (using random unconstrained $c$) for model training.

## Module 2: Target Defense Model (`mlp_nad.py`)
- Define a simple PyTorch MLP: `Input(m) -> Linear(128) -> ReLU -> Linear(64) -> ReLU -> Linear(1) -> Sigmoid`.
- Write a short training loop to train this MLP on the data generated in Module 1. It should classify Normal (1) vs. FDIA (0).
- Expose an inference function `query_black_box(z_prime)` that returns the float probability. This simulates the strict black-box feedback.

## Module 3: Projected ZOO Optimizer (`p_zoo.py`)
Implement the P-ZOO algorithm to find the optimal $c$.
- **Initialization**: $c = 0$.
- **Gradient Estimation (Semi-White-Box)**:
  - For the neural network part $\mathcal{N}(\cdot)$, use symmetric finite difference (coordinate-wise) to estimate the gradient vector $g_{NN} = \hat{\nabla}_c \mathcal{N}$. (Since IEEE 14 has only 14 nodes, you can do this for all $n$ dimensions without sparsity reduction for now).
  - For the graph regularization part, the analytical gradient is $g_{reg} = 2 \lambda L c$.
  - Total gradient: $g = g_{NN} - g_{reg}$.
- **Adam Update**:
  - Update a temporary variable $c_{temp} = c + \eta \cdot \text{Adam\_step}(g)$ (Note: we use $+$ because we are MAXIMIZING $J(c)$).
- **Projection Operator (QP Solver)**:
  - Project $c_{temp}$ back into the physical feasible region.
  - Solve: $\min_{c} \frac{1}{2} c^T c - c_{temp}^T c$ subject to $A_{ineq} c \le b_{ineq}$.
  - Use `scipy.optimize.minimize` (method='SLSQP') or a dedicated QP solver like `cvxopt` / `osqp`.
  - Update $c$ with the solved projected value.
- Iterate for a fixed number of steps (e.g., 100) or until $\mathcal{N}(z+Hc) > 0.95$.

## Module 4: Evaluation Pipeline (`main.py`)
- Load a test sample $z_{test}$.
- Run the P-ZOO attack to get the adversarial perturbation $c_{adv}$.
- Print the metrics:
  - Initial MLP score vs. Final MLP score (ASR).
  - Check if any physical limits are violated: `any(abs(M @ (x + c_adv)) > F_max)`.
  - Compute the graph smoothness score: $c_{adv}^T L c_{adv}$.

# Coding Guidelines for the Agent
- Use strict Python typing (`typing` module).
- Add detailed docstrings for the matrices ($H, M, L$) to ensure dimensions are tracked carefully.
- Ensure the projection QP problem is formulated correctly. If `scipy` is too slow, set up `osqp` matrices.
- Start by providing the code for **Module 1** only.