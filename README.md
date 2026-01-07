# Quantum Machine Learning GANs  

## 📌 Overview

This project explores **Generative Adversarial Networks** from both a **classical machine learning** and a **quantum machine learning** perspective.  
The long-term goal is to **design, implement, and compare**:

1. A **Classical GAN**
2. A **Quantum GAN**
3. A **Quantum CycleGAN or Hybrid Classical–Quantum GAN**

By comparing these approaches, the project aims to investigate whether **quantum circuits can offer advantages** in generative modeling, such as improved expressivity, efficiency, or learning dynamics.


## 🎯 Objectives

- Implement and train a **baseline Classical GAN**
- Develop a **Quantum GAN** using parameterized quantum circuits
- Compare classical and quantum models using:
  - Training stability
  - Sample quality
  - Convergence behavior
- Explore advanced architectures:
  - **Quantum CycleGAN**
  - **Hybrid Classical–Quantum GAN**

## 🧠 Background

Generative Adversarial Networks consist of two competing models:

- **Generator (G)**: Generates synthetic data
- **Discriminator (D)**: Distinguishes real data from generated data

In a **Quantum GAN**, one or both of these components are replaced with **quantum circuits**, leveraging:
- Quantum superposition
- Entanglement
- Variational quantum algorithms

This project is part of a broader exploration of **Quantum Machine Learning** and **Noisy Intermediate-Scale Quantum** devices.


## 🏗️ Structure

```text
.
├── classical_gan/
│   ├── model.py
│   ├── train.py
│   ├── utils.py
│   └── results/
│
├── quantum_gan/              # 🚧 In progress
│   ├── quantum_generator.py
│   ├── quantum_discriminator.py
│   └── circuits/
│
├── hybrid_gan/               # 🚧 Planned
│
├── data/
├── experiments/
├── notebooks/
└── README.md
```

## 🚀 Status: Classical & Quantum GANs

### 📌 Current Status

| Component            | Status |
|----------------------|--------|
| Classical GAN        | ✅ Implemented & Trained |
| Quantum GAN          | 🚧 In Progress |
| Quantum CycleGAN     | ⏳ Planned |
| Hybrid GAN           | ⏳ Planned |

---

### ⚠️ Important Notice

At the current stage of the project:

- **Only the Classical GAN** has been fully implemented and trained.
- **Quantum and hybrid approaches** are under active development and experimentation.


## 🔬 Classical GAN (Implemented)

- Fully classical neural network architecture  
- Serves as a **baseline reference**  
- Successfully trained and evaluated  
- Provides metrics for comparison with future quantum models  

---

## ⚛️ Quantum GAN (In Progress)

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

---

## 🔄 Quantum CycleGAN / Hybrid GAN (Planned)

### Potential Research Directions

- Quantum generator with classical discriminator  
- Bidirectional mappings using quantum circuits  
- Hybrid classical–quantum adversarial training loops  

## 📊 Evaluation Metrics (Planned)

- Generator and discriminator losses  
- Sample diversity and quality  
- Training stability  
- Computational overhead  
- Scalability  

## 🛠️ Technologies & Tools

- PyTorch / TensorFlow (Classical GAN)  
- PennyLane / Qiskit (Quantum components) 

## 📌 Future Work

- Complete Quantum GAN implementation  
- Perform systematic classical vs. quantum comparisons  
- Extend to CycleGAN and hybrid architectures  
- Evaluate performance on real quantum hardware (if feasible)  

## 📚 References

- I. Goodfellow et al., *Generative Adversarial Networks*  
- S. Lloyd et al., *Quantum Generative Adversarial Learning*  
- Quantum Machine Learning research literature  



## 📄 License

This project is licensed under the **APACHE 2.0 License**.