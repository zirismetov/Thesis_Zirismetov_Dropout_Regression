import torch
from torch.autograd import Variable

class Dropout(torch.nn.Module):
    def __init__(self, drop_p, layers_size, l):
        super().__init__()
        self.p = drop_p
        if self.p < 1:
            self.alpha = self.p/(1-self.p)
        else:
            self.alpha = 0.0

    def forward(self, x):

        if self.train():
            epsilon = torch.randn(x.size()) * self.alpha + 1

            epsilon = Variable(epsilon)
            if x.is_cuda:
                epsilon = epsilon.cuda()

            return x * epsilon
        else:
            return x