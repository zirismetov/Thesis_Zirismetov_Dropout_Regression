import torch
import numpy as np
# class MLP(torch.nn.Module):
#     def __init__(self, d_prob = [0 ,0.5 ,0.5, 0.5, 0.5], layers='last', layer_size=[1, 32, 32, 32, 1]):
#         super().__init__()
#
#         self.layers = torch.nn.Sequential()
#         for l in range(len(layer_size) - 2):
#             self.layers.add_module(f'linear_layer_{l + 1}',
#                                    torch.nn.Linear(int(layer_size[l]), int(layer_size[l + 1])))
#             self.layers.add_module(f'LeakyReLU_layer_{l + 1}',
#                                    torch.nn.LeakyReLU())
#             if layers != 'last':
#                 self.layers.add_module(f'SimpleDropout_layer_{l + 1}',
#                                        torch.nn.Dropout(p=prob))
#         if layers == 'last':
#             self.layers.add_module(f'SimpleDropout_layer_end',
#                                    torch.nn.Dropout(p=prob))
#         self.layers.add_module("last_linear_layer",
#                                torch.nn.Linear(int(layer_size[-2]), int(layer_size[-1])))
#
#     def forward(self, x):
#         y_prim = self.layers.forward(x)
#         return y_prim

class SimpleDropout(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        # multiplier is 1/(1-p). Set multiplier to 0 when p=1 to avoid error...
        if self.p < 1:
            self.multiplier_ = 1.0 / (1.0 - p)
        else:
            self.multiplier_ = 0.0

    def forward(self, input):
        # if model.eval(), don't apply dropout
        if not self.training:
            return input

        # So that we have `input.shape` numbers of Bernoulli(1-p) samples
        selected_ = torch.Tensor(input.shape).uniform_(0, 1) > self.p

        # Multiply output by multiplier as described in the paper [1]
        return torch.mul(selected_, input) * self.multiplier_