import numpy as np
import torch

class LinearLayer(torch.nn.Module):
    def __init__(self, input_size:int, output_size:int, gain:float=2**(0.5), use_wscale:bool=False, lrmul:float=1, bias:bool=True):
        super().__init__()
        he_std = gain * input_size**(-0.5)
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(torch.randn(output_size, input_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_size))
            self.b_mul = lrmul
        else:
            self.bias = None

    def forward(self, x:np.ndarray):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
        return torch.nn.functional.linear(x, self.weight * self.w_mul, bias)