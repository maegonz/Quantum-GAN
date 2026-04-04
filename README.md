# Quantum GANs  

## Objectives

The objectives of this project is to develop a **quantum GAN** using parameterized quantum circuits in order to be able to compare both classical and quantum models and study *training stability*, *sample quality* or *convergence behavior*.

This project is part of a broader exploration of **Quantum Computing** and **Quantum Machine Learning**.

## Background

Generative Adversarial Networks consist of two competing models:

- **Generator (G)**: Generates synthetic data
- **Discriminator (D)**: Distinguishes real data from generated data

The objective function is describe as follows:

$$\min_{G} \max_{D} V(D, G) = \mathbb{E}_{x \sim p_{data}} [\log D(x)] + \mathbb{E}_{z \sim p_{z}} [\log(1 - D(G(z)))]$$

One objective of G is to minimize V, whereas another objective of G is to maximize V.

In this project, an **hybrid architecture** is opted for, as mentionned in most of research papers. Specifically, the generator is based on a variational quantum algorithm and the discriminator is a classic CNN, even though, both generator and discriminator could rely on quantum algorithms, leveraging *quantum superposition* and *entanglement*.



## Structure

```text
.
├── data/
│   └── dataset.py
│
├── qgan/              
│   ├── generator.py
│   ├── discriminator.py
│   └── vqc.py
│
├── experiments/
├── main.ipynb
├── LICENSE
└── README.md
```

## References

- J. Jäger et al., *Scaling Quantum Machine Learning without Tricks: High-Resolution and Diverse Image Generation* [2026]
- Riofrio et al., *A Characterization of Quantum Generative Models* [2024]
- Benedetti et al., *Parameterized quantum circuits as machine learning models* [2019]
- I. Goodfellow et al., *Generative Adversarial Networks* [2014]
- S. Lloyd et al., *Quantum Generative Adversarial Learning* [2018]
- M. Cerezo et al. *Variational Quantum Algorithms* [2021]
- J.McClean et al. *Barren plateus in quantum neural networke training landscapes*

This list is not exhaustive, is subject to change.


## License

This project is licensed under the **APACHE 2.0 License**.
