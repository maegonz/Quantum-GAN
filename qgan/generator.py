import pennylane as qml
import torch
import torch.nn as nn
from pennylane import numpy as np
from .vqc import quantum_circuit, patch_circuit


# class QuantumGenerator(nn.Module):
#     def __init__(self, n_qubits=4, n_layers=2, circuit=quantum_circuit):
#         super().__init__()
#         self.qubits = n_qubits  #Represents pixels
#         self.n_layers = n_layers

#         # Initializing weights for the RY, RX, RZ gates per layer/qubit
#         self.weights = nn.Parameter(0.1 * torch.randn(n_layers, n_qubits, 3))
        
#         self.dev = qml.device("default.qubit", wires=self.qubits)
#         self.qnode_circuit = qml.QNode(quantum_circuit, self.dev, interface="torch")

#     def forward(self, x):
#         # x is the latent noise vector
#         # PennyLane handles the gradient through the QNode automatically
#         out = torch.stack([self.qnode_circuit(self.qubits, self.n_layers, i, self.weights) for i in x])
#         # Rescale from [-1, 1] to [0, 1] for MNIST pixels
#         return (out + 1) / 2

class PatchQuantumGenerator(nn.Module):
    def __init__(self, n_qubits=4, n_patches=49, patches_size=4, circuit=patch_circuit, device=None):
        super().__init__()
        self.qubits = n_qubits
        self.patch_size = patches_size
        self.circuit = circuit
        self.n_patches = n_patches
        # Each patch gets its own set of 3 layers of weights
        self.patch_weights = nn.Parameter(
            torch.randn(n_patches, 3, n_qubits, 3, device=device) * 0.01 # (patches, layers, qubits, rotations)
        )

        self.dev = qml.device("default.qubit", wires=self.qubits)
        self.qnode_circuit = qml.QNode(patch_circuit, self.dev, interface="torch")  # Adjoint differentation for faster gradients calculation


    def _reconstruct_image(self, patches):
        # Logic to turn 49 patches of 16 pixels into a 28x28 tensor
        # batch, 49, 16 -> batch, 1, 28, 28
        batch_size = patches.shape[0]
        # First, reshape 16 to 4x4
        patches = patches.view(batch_size, 7, 7, 4, 4)
        # Permute and reshape to combine grid
        images = patches.permute(0, 1, 3, 2, 4).contiguous()
        return images.view(batch_size, 1, 28, 28)
    
    def forward(self, noise_batch):
        # noise_batch shape: (batch_size, n_qubits)
        final_image = []

        for p in range(self.n_patches):
            # Generate one patch for the entire batch
            patch = self.qnode_circuit(self.qubits, noise_batch, self.patch_weights[p])
            final_image.append(patch)

        # Combine patches: (n_patches, batch, 16) -> (batch, n_patches, 16)
        combined = torch.stack(final_image, dim=1)
        
        # Reconstruct 28x28 image logic
        # This requires reshaping and 'unfolding' the patches back into a grid
        return self._reconstruct_image(combined)
