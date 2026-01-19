import torch.nn as nn
from .blocks import GenBlock

## Generator network for a Deep Convolutional GAN (DCGAN)
class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels, num_features=64, bias=True):
        """
        Initialize the Generator network.
        
        Params
        -------
        latent_dim : int
            Dimension of the input latent nois vector.
        img_channels : int
            Number of channels in the output image.
        num_features : int, Optional
            Number of features in the first convolutional layer.
            Defaults to 64.
        bias : bool, Optional
            Whether to use random bias or custom bias in the convolution layers.
            Defaults to True.
        """
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.generator = nn.Sequential(
            # From latent vector Z to feature map
            GenBlock(latent_dim, num_features * 8, stride=1, bias=bias),
            # Upsampling layers to increase spatial dimension
            # (num_features*8) x 4 x 4
            GenBlock(num_features * 8, num_features * 4, bias=bias),
            # (num_features*4) x 8 x 8
            GenBlock(num_features * 4, num_features * 2, bias=bias),
            # (num_features*2) x 16 x 16
            GenBlock(num_features * 2, num_features, bias=bias),
            # (num_features) x 32 x 32
            nn.ConvTranspose2d(num_features, img_channels, kernel_size=4, stride=2, padding=1, bias=bias),
            nn.Tanh()  # Output activation to map values between -1 and 1
        )
    
    def forward(self, input):
        """
        Note: input is a noise vector of shape (batch_size, latent_dim, 1, 1)
        """
        output = self.generator(input)
        return output