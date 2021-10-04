import torch


class SimpleDropout(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        # multiplier is 1/(1-p). Set multiplier to 0 when p=1 to avoid error...
        if self.p < 1:
            self.multiplier = 1.0 / (1.0 - p)
        else:
            self.multiplier = 0.0

    def forward(self, input):
        # if model.eval(), don't apply dropout
        if not self.training:
            return input

        # So that we have `input.shape` numbers of Bernoulli(1-p) samples
        selected = torch.Tensor(input.shape).uniform_(0, 1) > self.p

        # Multiply output by multiplier as described in the paper [1]
        return torch.mul(selected, input) * self.multiplier
