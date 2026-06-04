import torch
import torch.nn as nn

class QLIKELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(QLIKELoss, self).__init__()
        self.eps = eps

    def forward(self, pred, actual):
        ratio = actual / (pred + self.eps)
        loss = ratio - torch.log(ratio + self.eps) - 1

        return torch.mean(loss)