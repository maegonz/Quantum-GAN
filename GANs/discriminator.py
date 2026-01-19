import torch.nn as nn
from .blocks import DiscBlock

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, num_features=64, bias=True):
        """ 
        Initialize the Discriminator network.

        Params
        -------
        in_channels : int, Optional
            Number of input channels.
            Defaults to 3.
        num_features : int, Optional
            Base number of features for the convolutional layers.
            Defaults to 64.
        bias : bool, Optional
            Whether to use bias in the convolution layers.
            Defaults to True.
        """
        super(Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
            # Input: in_channels x 64 x 64
            nn.Conv2d(in_channels, num_features, kernel_size=4, stride=2, padding=1, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            # num_features x 32 x 32
            DiscBlock(num_features, num_features * 2, bias=bias),
            # (num_features*2) x 16 x 16
            DiscBlock(num_features * 2, num_features * 4, bias=bias),
            # (num_features*4) x 8 x 8
            DiscBlock(num_features * 4, num_features * 8, bias=bias),
            # (num_features*8) x 4 x 4
            nn.Conv2d(num_features * 8, 1, kernel_size=4, stride=1, padding=0, bias=bias),
            # Output: 1 x 1 x 1
            nn.Sigmoid()
        )
    
    def forward(self, input):
        output = self.discriminator(input)
        return output.view(-1, 1).squeeze(1)