import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn

def quantum_circuit(n_qubits, n_layers, noise, weights):
    # 1. Latent Space Mapping (Encoding Noise)
    for i in range(n_qubits):
        qml.RY(noise[i], wires=i)
    
    # 2. Trainable Variational Layers
    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RX(weights[l, i, 0], wires=i)
            qml.RY(weights[l, i, 1], wires=i)
            qml.RZ(weights[l, i, 2], wires=i)
        
        # Entanglement (Crucial for spatial correlation in images)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
            
    # 3. Measurement (Returns expectation values as pixel intensities)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

def patch_circuit(n_qubits, noise, weights):
    # Encoding: Transform noise into quantum state
    # We use Angle Embedding for simplicity here
    qml.AngleEmbedding(noise, wires=range(n_qubits))
    
    # Variational Layers (The "Brain" of the generator)
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    
    # Return probabilities (16 states for 16 pixels)
    return qml.probs(wires=range(n_qubits))