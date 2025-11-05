# Dendritic Local Learning vs Backpropagation  
### How Far Can a Single Neuron Go on Nonlinear Tasks

---

## Objective
Investigate whether a **single neuron with nonlinear dendrites**, trained by a **local three-factor learning rule**, can perform nonlinear computations such as **XOR** and **parity**, and how its behavior compares to:

1. The same neuron trained with **backpropagation**
2. A small **two-layer MLP** trained with backpropagation

---

## Motivation
Backpropagation is the dominant algorithm for training artificial neural networks but is **biologically implausible**, relying on global error signals and symmetric weight transport.  
In contrast, biological neurons learn through **local dendritic computations** and **three-factor rules** (pre-synaptic input × dendritic activity × modulatory signal).  
This project explores how far such local mechanisms can go in solving nonlinear tasks compared to global gradient-based learning.

---

## Research Questions
- Can dendritic local learning achieve nonlinear separability on XOR and parity tasks?  
- How does its **sample efficiency**, **stability**, and **robustness** compare to backpropagation?  
- How does increasing **dendritic subunit count** affect expressivity and performance?

---

## Hypotheses
- **H1:** A dendritic neuron trained by a local three-factor rule can solve nonlinear tasks that a linear perceptron cannot.  
- **H2:** Dendritic learning achieves comparable or better **sample efficiency** than backpropagation on small nonlinear datasets.  
- **H3:** **Expressivity** increases with dendritic branch count but eventually saturates.

---

## Method Overview
- Implement a **two-compartment neuron** (dendrites + soma) using PyTorch or NumPy.  
- Compare three models:
  - Linear perceptron (backprop)
  - Dendritic neuron (local rule and backprop)
  - Two-layer MLP (backprop)
- Apply a **three-factor local update rule**:

  \[
  \Delta w_{ij} = \eta \, x_i \, f'(u_j) \, (r - \hat{r})
  \]

  where \(x_i\) is input to dendrite \(j\), \(f'(u_j)\) is the local activation derivative, and \((r - \hat{r})\) is a modulatory error signal.

- Tasks: XOR, 3-bit parity, optional 4-bit parity  
- Metrics: accuracy, convergence speed, sample efficiency, robustness to input noise

---

## Experiments

| Experiment | Variable | Measure |
|-------------|-----------|----------|
| **A1** | Dendritic vs backprop on XOR | Accuracy, decision boundaries |
| **A2** | Add Gaussian noise | Robustness to input noise |
| **B1** | Vary dendrite count (1–5) | Expressivity (max parity task solved) |
| **B2** | Reduce training samples | Sample efficiency |

---

## Deliverables
- **10-page NeurIPS-style paper**
  - ~5 pages literature review on dendritic and local learning
  - ~5 pages experiments, analysis, and discussion  
- **Code notebook:** documented implementation and plots  
- **Figures:** decision boundaries, learning curves, dendrite-count ablation  
- **Summary table:** comparative performance metrics

---

## Expected Outcome
Demonstrate that **nonlinear dendritic structure + local learning** can achieve nonlinear separability and sample-efficient learning without global backpropagation — highlighting dendrites as a potential computational alternative to multilayer backprop-trained networks.
