import pennylane as qml
import torch
import torch.nn as nn
import jax
import jax.numpy as jnp
import flax.linen as nn
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

# class PatchQuantumGenerator(nn.Module):
#     def __init__(self, n_qubits=4, n_patches=49, patches_size=4, circuit=patch_circuit, device=None):
#         super().__init__()
#         self.qubits = n_qubits
#         self.patch_size = patches_size
#         self.circuit = circuit
#         self.n_patches = n_patches
#         # Each patch gets its own set of 3 layers of weights
#         self.patch_weights = nn.Parameter(
#             torch.randn(n_patches, 3, n_qubits, 3, device=device) * 0.01 # (patches, layers, qubits, rotations)
#         )

#         self.dev = qml.device("default.qubit", wires=self.qubits)
#         self.qnode_circuit = qml.QNode(patch_circuit, self.dev, interface="torch")  # Adjoint differentation for faster gradients calculation


#     def _reconstruct_image(self, patches):
#         # Logic to turn 49 patches of 16 pixels into a 28x28 tensor
#         # batch, 49, 16 -> batch, 1, 28, 28
#         batch_size = patches.shape[0]
#         # First, reshape 16 to 4x4
#         patches = patches.view(batch_size, 7, 7, 4, 4)
#         # Permute and reshape to combine grid
#         images = patches.permute(0, 1, 3, 2, 4).contiguous()
#         return images.view(batch_size, 1, 28, 28)
    
#     def forward(self, noise_batch):
#         # noise_batch shape: (batch_size, n_qubits)
#         final_image = []

#         for p in range(self.n_patches):
#             # Generate one patch for the entire batch
#             patch = self.qnode_circuit(self.qubits, noise_batch, self.patch_weights[p])
#             final_image.append(patch)

#         # Combine patches: (n_patches, batch, 16) -> (batch, n_patches, 16)
#         combined = torch.stack(final_image, dim=1)
        
#         # Reconstruct 28x28 image logic
#         # This requires reshaping and 'unfolding' the patches back into a grid
#         return self._reconstruct_image(combined)


class Generator(nn.Module):
    @nn.compact
    def __call__(self,
                 circuit: function,
                 weights: jnp.ndarray,
                 noise: jnp.ndarray,
                 n_qubits: int=4,
                 n_patches: int=49):
        """
        Quantum generator that produces 28x28 images by generating 49 patches of 4x4 pixels each using a quantum circuit.
        Each patch is generated from the same noise vector but with different trainable weights, allowing for spatial correlation in the final image.

        Parameters
        ----------
        circuit: function
            The quantum circuit function to generate patches.
        weights: jnp.ndarray
            The trainable weights for the quantum circuit, shape (n_patches, layers, qubits, rotations).
        noise: jnp.ndarray
            The input noise vector for the generator, shape (batch_size, n_qubits).
        n_qubits: int
            The number of qubits (and thus pixels per patch), defaults to 4.
        n_patches: int
            The number of patches to generate, defaults to 49 for 28x28 images with 4x4 patches.
        
        Returns
        -------
        generated_image: jnp.ndarray
            The generated images tensor with shape (batch_size, 28, 28, 1) NCHW format.
        """

        # Function that generates 49 from a single noise vector
        patched_images = jax.vmap(circuit, in_axes=(None, 0))  # Map over axis 0 of weights not map noise

        # Function that runs the above over a batch of noise vectors
        batch_patched_images = jax.vmap(patched_images, in_axes=(0, None))  # Map over noise batch, not weights

        # Generate patches for the entire batch
        dispatched_images = batch_patched_images(weights, noise)  # (batch_size, n_patches, 16)

        generated_images = self._reconstruct_image(dispatched_images)
        
        return generated_images

    def _reconstruct_image(self, patches: jnp.ndarray):
        """
        Reconstructs a 28x28 image from 49 patches of 4x4 pixels each.
        The patches are arranged in a 7x7 grid to form the final image.

        Parameters
        ----------
        patches: jnp.ndarray
            The generated patches tensor with shape (batch_size, n_patches, 16) where 16 corresponds to the 4x4 pixels of each patch.

        Returns
        -------
        images: jnp.ndarray
            The reconstructed images tensor with shape (batch_size, 28, 28, 1) NHWC format.
        """

        batch_size = patches.shape[0]

        # Reshape 16 pixels into 4x4 patches and arrange into 7x7 grid
        patches = patches.reshape(batch_size, 7, 7, 4, 4)

        # Permute and reshape to combine grid
        # Current axes: 0:batch, 1:grid_y, 2:grid_x, 3:patch_y, 4:patch_x
        # Target axes:  0:batch, 1:grid_y, 3:patch_y, 2:grid_x, 4:patch_x
        images = jnp.transpose(patches, (0, 1, 3, 2, 4))  # (batch_size, 7, 4, 7, 4)
        
        # Combine grid and patch dimensions into final image shape
        images = images.reshape((batch_size, 28, 28, 1))  
        return images