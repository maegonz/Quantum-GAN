import pennylane as qml
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pennylane import numpy as np


class Generator(nn.Module):
    def __init__(self,
                 circuit,
                 n_subgenerator: int=16,
                 n_layers: int=3,
                 n_qubits: int=5, 
                 n_ancillas: int=1,
                 img_shape: int=16,
                 device=None):
        super().__init__()

        self.circuit = circuit
        self.n_subgenerator = n_subgenerator
        self.n_layers = n_layers
        self.n_qubits = n_qubits
        self.n_ancillas = n_ancillas
        self.patch_size = 2**(self.n_qubits - self.n_ancillas)
        self.img_shape = img_shape

        # Each subgenerator has its own set of weights for the quantum circuit
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(self.n_layers, self.n_qubits, 3, device=device)) for _ in range(self.n_subgenerator)
        ])

        self.dev = qml.device("default.qubit", wires=range(self.n_qubits))
        self.qnode = qml.QNode(self.circuit, self.dev, interface="torch")  # Adjoint differentation for faster gradients calculation

    
    def forward(self, noise_batch):
        batch_size = noise_batch.size(0)
        subgenerator_outputs = []

        for weight in self.weights:
            # Shape: (batch_size, 2**n_qubits)
            probs = self.qnode(weight, noise_batch, n_qubits=self.n_qubits, a_qubits=self.n_ancillas, n_layers=self.n_layers)  
            # Shape: (batch_size, patch_size)
            probs_ancilla_0 = probs[:, :self.patch_size]

            # Normalize
            sum_probs = torch.sum(probs_ancilla_0, dim=1, keepdim=True) + 1e-8
            probs_ancilla_0 = probs_ancilla_0 / sum_probs

            max_prob, _ = torch.max(probs_ancilla_0, dim=1, keepdim=True)
            probs_ancilla_0 = 2 * ((probs_ancilla_0 / (max_prob + 1e-8)) - 0.5)
            
            subgenerator_outputs.append(probs_ancilla_0.float())

        # Shape: (batch_size, n_subgenerator, patch_size)
        output = torch.stack(subgenerator_outputs, dim=1)
                                  
        # Shape: (batch_size, 1, img_shape, img_shape)
        final_output = output.view(batch_size, 1, self.img_shape, self.img_shape)
        
        return final_output  