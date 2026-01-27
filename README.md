# Quantum GANs  

## Overview

This project explores **Generative Adversarial Networks** from both a **classical machine learning** and a **quantum machine learning** perspective.  
The long-term goal is to **design, implement, and compare**:

1. A **Classical GAN**
2. A **Quantum GAN**
3. A **Quantum CycleGAN or Hybrid Classical–Quantum GAN**

By comparing these approaches, the project aims to investigate whether **quantum circuits can offer advantages** in generative modeling, such as improved expressivity, efficiency, or learning dynamics.\
The major libraries used are PyTorch and TensorFlow for the classical GAN implementation and PennyLane and Qiskit for the quantum components implementation.


## Objectives

The objectives of this project is in the first place to implement and train a **baseline classical GAN**, then develop a **quantum GAN** using parameterized quantum circuits in order to be able to compare both classical and quantum models and study *training stability*, *sample quality* or *convergence behavior*.

Afterwards, I aim to explore more advanced architectures, such as **quantum Cycle GAN** and **hybrid classical-quantum GAN**.

## Background

Generative Adversarial Networks consist of two competing models:

- **Generator (G)**: Generates synthetic data
- **Discriminator (D)**: Distinguishes real data from generated data

The objective function is describe as follows:

$$\min_{G} \max_{D} V(D, G) = \mathbb{E}_{x \sim p_{data}} [\log D(x)] + \mathbb{E}_{z \sim p_{z}} [\log(1 - D(G(z)))]$$

One objective of G is to minimize V, whereas another objective of G is to maximize V.

In a **Quantum GAN**, one or both of these components are replaced with **quantum circuits**, leveraging:
- Quantum superposition
- Entanglement
- Variational quantum algorithms

This project is part of a broader exploration of **Quantum Machine Learning** and **Noisy Intermediate-Scale Quantum** devices.


## Structure

```text
.
├── data/
│   └── dataset.py
│
├── GANs/
│   ├── blocks.py
│   ├── discriminator.py
│   ├── generator.py
│   ├── methods.py
│   ├── utils.py
│   └── results/
│
├── QGANs/              # 🚧 In progress
│   ├── quantum_generator.py
│   ├── quantum_discriminator.py
│   └── circuits/
│
├── Hybrid_GANs/               # 🚧 Planned
│
├── experiments/
├── notebooks/
├── main.py
├── LICENSE
└── README.md
```


### ⚠️ Important Notice

At the current stage of the project:

- **Only the Classical GAN** has been fully implemented and trained.
- **Quantum and hybrid approaches** are under active development and experimentation.

| Component            | Status |
|----------------------|--------|
| Classical GAN        | ✅ Implemented & Trained |
| Quantum GAN          | 🚧 In Progress |
| Quantum CycleGAN     | ⏳ Planned |
| Hybrid GAN           | ⏳ Planned |


<!-- ## Classical GAN

- Fully classical neural network architecture  
- Serves as a **baseline reference**  
- Successfully trained and evaluated  
- Provides metrics for comparison with future quantum models   -->


## Quantum GAN

### Planned Characteristics

- Parameterized Quantum Circuits (PQCs)  
- Quantum generator and/or discriminator  
- Hybrid training with classical optimizers  
- Simulation using quantum frameworks:
  - PennyLane
  - Qiskit

---

### Challenges Encountered

#### Barren Plateaus

A primary challenge in the training of PQDs is the presence of **barren plateaus**, characterized by regions of the loss landscape where gradients vanish exponentially with increasing system size or circuit depth. This phenomenon significantly hinders the effectiveness of gradient-based optimization methods.

**Underlying causes**

* Random or unstructured parameter initialization
* Highly expressive and deep variational ansätze
* The use of global cost functions acting on many qubits

**Consequences**

* Exponentially small gradients impede parameter updates
* Training becomes inefficient or stalls entirely
* Classical optimizers fail to converge to meaningful solutions

In practice, barren plateaus manifest as early stagnation during training, with minimal observable improvement in the objective function.

---

#### Noise Sensitivity

Parameterized quantum models are highly sensitive to noise arising from decoherence, gate imperfections, and measurement errors. This sensitivity persists even in simulated environments when realistic noise models are incorporated, reflecting limitations of near-term quantum devices.

**Underlying causes**

* Absence of fault-tolerant error correction in NISQ-era systems
* Accumulation of noise with increasing circuit depth
* Amplification of errors through repeated variational updates

**Consequences**

* Reduced fidelity of quantum states and outputs
* Increased variance in expectation value estimates
* Degradation of training stability and model performance

Empirically, this sensitivity is observed as a marked performance gap between noiseless and noisy simulations, as well as reduced reproducibility across independent training runs.

---

#### Gradient Estimation

Training variational quantum models requires the estimation of gradients using techniques such as the **parameter-shift rule**, which introduces substantial computational and statistical overhead.

**Underlying causes**

* Each gradient component necessitates multiple circuit evaluations
* Finite sampling (shot noise) affects gradient accuracy
* Gradient evaluation cost scales poorly with the number of parameters

**Consequences**

* Increased computational resource requirements
* Noisy gradients slow convergence and destabilize optimization
* Strong dependence on optimizer selection and hyperparameter tuning

In practice, reliable gradient estimation often requires a large number of measurement shots, leading to a trade-off between computational efficiency and optimization accuracy.

---

### Interdependence of Challenges

These challenges are strongly interrelated. Noise can exacerbate barren plateau effects by further diminishing gradient magnitudes, while inaccurate gradient estimates can effectively flatten the optimization landscape. Consequently, the successful training of PQCs necessitates careful circuit design, informed initialization strategies, and robust hybrid quantum–classical optimization approaches.


## Evaluation Metrics

- Generator and discriminator losses  
- Sample diversity and quality  
- Training stability  
- Computational overhead  
- Scalability  

## Potential Research Directions & Future Work

- Quantum generator with classical discriminator  
- Bidirectional mappings using quantum circuits  
- Hybrid classical–quantum adversarial training loops  
- Perform systematic classical vs. quantum comparisons  
- Extend to CycleGAN and hybrid architectures  
- Evaluate performance on real quantum hardware (if feasible)  

## References

- I. Goodfellow et al., *Generative Adversarial Networks*  
- S. Lloyd et al., *Quantum Generative Adversarial Learning*
- M. Cerezo et al. *Variational Quantum ALgorithms*
- J.McClean et al. *Barren plateus in quantum neural networke training landscapes*

This list is not exhaustive, is subject to change.


## License

This project is licensed under the **APACHE 2.0 License**.
