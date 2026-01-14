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

- Implement and train a **baseline Classical GAN**
- Develop a **Quantum GAN** using parameterized quantum circuits
- Compare classical and quantum models using:
  - Training stability
  - Sample quality
  - Convergence behavior
- Explore advanced architectures:
  - **Quantum CycleGAN**
  - **Hybrid Classical–Quantum GAN**

## Background

Generative Adversarial Networks consist of two competing models:

- **Generator (G)**: Generates synthetic data
- **Discriminator (D)**: Distinguishes real data from generated data

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


## Classical GAN

- Fully classical neural network architecture  
- Serves as a **baseline reference**  
- Successfully trained and evaluated  
- Provides metrics for comparison with future quantum models  


## Quantum GAN

### Planned Characteristics

- Parameterized Quantum Circuits (PQCs)  
- Quantum generator and/or discriminator  
- Hybrid training with classical optimizers  
- Simulation using quantum frameworks:
  - PennyLane
  - Qiskit

### Challenges Being Explored

- Barren plateaus  
- Noise sensitivity  
- Gradient estimation  

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

This list is not exhaustive, is subject to change.



## License

This project is licensed under the **APACHE 2.0 License**.
