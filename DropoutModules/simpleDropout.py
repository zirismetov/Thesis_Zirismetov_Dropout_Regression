import torch




class Dropout(torch.nn.Module):
    def __init__(self, drop_p, layers_size, l):
        super().__init__()
        self.p = drop_p
        # multiplier is 1/(1-p). Set multiplier to 0 when p=1 to avoid error...
        if self.p < 1:
            self.multiplier = 1.0 / (1.0 - self.p)
        else:
            self.multiplier = 0.0

        if torch.cuda.is_available():
            self.DEVICE = "cuda"
        else:
            self.DEVICE = "cpu"

    def forward(self, input):
        # if model.eval(), don't apply dropout
        if not self.training:
            return input

        # So that we have `input.shape` numbers of Bernoulli(1-p) samples
        selected = (torch.Tensor(input.shape).uniform_(0, 1) > self.p).to(self.DEVICE)
        res = torch.mul(selected, input)
        out = res * self.multiplier
        # Multiply output by multiplier as described in the paper [1]
        return out
