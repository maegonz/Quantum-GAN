# import torch
# import torch.nn as nn

# class Discriminator(nn.Module):
#     def __init__(self, in_channels=1, num_features=64, bias=True):
#         """ 
#         Initialize the Discriminator network.

#         Params
#         -------
#         in_channels : int, Optional
#             Number of input channels.
#             Defaults to 1.
#         num_features : int, Optional
#             Base number of features for the convolutional layers.
#             Defaults to 64.
#         bias : bool, Optional
#             Whether to use bias in the convolution layers.
#             Defaults to True.
#         """
#         super(Discriminator, self).__init__()

#         self.discriminator = nn.Sequential(
#             # Input: (batch, 1, 28, 28)
#             nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1), # -> (batch, 64, 14, 14)
#             leaky_relu(negative_slop=0.2),
            
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> (batch, 128, 7, 7)
#             # InstanceNorm is preferred over BatchNorm for WGAN-GP
#             nn.InstanceNorm2d(128),
#             leaky_relu(negative_slop=0.2),
            
#             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # -> (batch, 256, 4, 4)
#             nn.InstanceNorm2d(256),
#             leaky_relu(negative_slop=0.2),
            
#             nn.Flatten(),
#             nn.Dropout(0.3),
#             # Final Layer: No Sigmoid! Output is a "Realness Score"
#             nn.Linear(256 * 4 * 4, 1)
#         )
    
#     def forward(self, input):
#         output = self.discriminator(input)
#         return output

import flax.linen as nn
import jax.numpy as jnp

class Discriminator(nn.Module):
    # This replaces __init__. Parameters are defined as class attributes.
    num_features: int = 64

    @nn.compact
    def __call__(self, x, training: bool = True):
        # x expected shape: (batch, 28, 28, 1)
        
        # Block 1
        x = nn.Conv(features=self.num_features, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        
        # Block 2
        x = nn.Conv(features=self.num_features * 2, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        # x = nn.GroupNorm(num_groups=self.num_features * 2)(x)
        x = nn.LayerNorm()(x)  # LayerNorm can be used as an alternative to GroupNorm 
        # x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.relu(x)         # ReLU can be used as an alternative to LeakyReLU
        
        # Block 3
        x = nn.Conv(features=self.num_features * 4, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        # x = nn.GroupNorm(num_groups=self.num_features * 4)(x)
        x = nn.LayerNorm()(x)  # LayerNorm can be used as an alternative to GroupNorm
        # x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.relu(x)         # ReLU can be used as an alternative to LeakyReLU
        
        # Flatten for the final dense layer
        x = x.reshape((x.shape[0], -1))
        
        # Final Layer
        # Dropout in Flax requires a deterministic flag tied to the training state
        x = nn.Dropout(rate=0.3, deterministic=not training)(x)
        
        x = nn.Dense(features=1)(x)
        
        return x