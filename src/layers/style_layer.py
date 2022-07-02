import numpy as np
from src.layers.linear_layer import LinearLayer
import torch.nn as nn

class StyleLayer(nn.Module):
    def __init__(self, latent_size, channels, use_wscale):
        super(StyleLayer, self).__init__()
        self.lin = LinearLayer(latent_size,
                            channels * 2,
                            gain=1.0, use_wscale=use_wscale)
        
    def forward(self, x, latent):
        style = self.lin(latent) # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1)] + (x.dim() - 2) * [1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x