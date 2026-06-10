import torch
import torch.nn as nn
from torch.nn import LeakyReLU as leaky_relu

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, num_features=64, bias=True):
        """ 
        Initialize the Discriminator network.

        Params
        -------
        in_channels : int, Optional
            Number of input channels.
            Defaults to 1.
        num_features : int, Optional
            Base number of features for the convolutional layers.
            Defaults to 64.
        bias : bool, Optional
            Whether to use bias in the convolution layers.
            Defaults to True.
        """
        super(Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
            # Input: (batch, 1, 16, 16)
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1), # -> (batch, 64, 14, 14) 8, 8
            leaky_relu(negative_slope=0.2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> (batch, 128, 7, 7) 4, 4
            # InstanceNorm is preferred over BatchNorm for WGAN-GP
            nn.InstanceNorm2d(128),
            leaky_relu(negative_slope=0.2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # -> (batch, 256, 4, 4) 2, 2
            nn.InstanceNorm2d(256),
            leaky_relu(negative_slope=0.2),
            
            nn.Flatten(),
            nn.Dropout(0.3),
            # Final Layer: No Sigmoid! Output is a "Realness Score"
            # nn.Linear(256 * 4 * 4, 1)
            nn.Linear(256 * 2 * 2, 1)
        )
    
    def forward(self, input):
        output = self.discriminator(input)
        return output