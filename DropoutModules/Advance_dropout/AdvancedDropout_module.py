import torch
import numpy as np
from torch.nn.parameter import Parameter
import torch.nn.functional as F


# class Module_layers(torch.nn.Module):
#     def __init__(self, prob=0.5, layers='last', layer_size=[1, 32, 32, 32, 1]):
#         super().__init__()
#
#         self.layer_size = []
#         for i in layer_size:
#             self.layer_size.append(int(i))
#
#         self.layers = torch.nn.Sequential()
#         for l in range(len(layer_size) - 2):
#             self.layers.add_module(f'linear_layer_{l + 1}',
#                                    torch.nn.Linear(int(layer_size[l]), int(layer_size[l + 1])))
#             self.layers.add_module(f'LeakyReLU_layer_{l + 1}',
#                                    torch.nn.LeakyReLU())
#             if layers != 'last':
#                 self.layers.add_module(f'AdvanceDropout_layer_{l + 1}',
#                                        AdvancedDropout(self.layer_size[l]))
#         if layers == 'last':
#             self.layers.add_module(f'AdvanceDropout_layer_end',
#                                    AdvancedDropout(self.layer_size[-1]))
#
#         self.layers.add_module("last_linear_layer",
#                                torch.nn.Linear(int(layer_size[-2]), int(layer_size[-1])))
#
#     def forward(self, x):
#         y_prim = self.layers.forward(x)
#         return y_prim
#

class AdvancedDropout(torch.nn.Module):

    def __init__(self, num, init_mu=0, init_sigma=1.2, reduction=16):
        '''
        params:
        num (int): node number
        init_mu (float): intial mu
        init_sigma (float): initial sigma
        reduction (int, power of two): reduction of dimention of hidden states h
        '''
        super().__init__()
        if init_sigma <= 0:
            raise ValueError("Sigma has to be larger than 0, but got init_sigma=" + str(init_sigma))
        self.init_mu = init_mu
        self.init_sigma = init_sigma

        self.weight_h = Parameter(torch.rand([num // reduction, num]).mul(0.01))
        self.bias_h = Parameter(torch.rand([1]).mul(0.01))

        self.weight_mu = Parameter(torch.rand([1, num // reduction]).mul(0.01))
        self.bias_mu = Parameter(torch.Tensor([self.init_mu]))
        self.weight_sigma = Parameter(torch.rand([1, num // reduction]).mul(0.01))
        self.bias_sigma = Parameter(torch.Tensor([self.init_sigma]))

    def forward(self, input):
        if self.training:
            c, n = input.size()
            # parameterized prior
            h = F.linear(input, self.weight_h, self.bias_h)
            mu = F.linear(h, self.weight_mu, self.bias_mu).mean()
            sigma = F.softplus(F.linear(h, self.weight_sigma, self.bias_sigma)).mean()
            # mask
            epsilon = mu + sigma * torch.randn([c, n])
            mask = torch.sigmoid(epsilon)

            out = input.mul(mask).div(torch.sigmoid(mu.data / torch.sqrt(1. + 3.14 / 8. * sigma.data ** 2.)))
        else:
            out = input

        return out
