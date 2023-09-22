import torch
from torch import Tensor
from torch.nn import Module, MSELoss, CosineSimilarity


class MarginLoss(Module):
    def __init__(self, loss_fn: Module, margin: float = 0.0):
        super().__init__()
        self.loss_fn = loss_fn
        self.margin = margin

    def forward(self, embeds: Tensor, labels: Tensor):
        loss = []
        for shape_i in range(embeds.shape[0]):
            input_i = embeds[shape_i]
            label_i= labels[shape_i]
            for shape_j in range(embeds.shape[0]):
                input_j = embeds[shape_j]
                label_j = labels[shape_j]
                d = self.loss_fn(input_i, input_j)
                if label_i == label_j:
                    loss.append(1-d)
                else:
                    loss.append(torch.clip((d - self.margin), min=0.))
        return torch.mean(torch.stack(loss))


class ContrastLoss(Module):

    def __init__(self, loss_fn: Module, margin: float = 0.0):
        super().__init__()
        self.margin = margin
        self.loss_fn = loss_fn

    def forward(self, pred1: Tensor, pred2: Tensor, labels: Tensor):
        # input: (B, C, T)
        loss = []
        for i in range(pred1.shape[0]):
            # mean L2 distance squared
            d = self.loss_fn(pred1[i, :], pred2[i, :])
            # d = self.cosim(pred1[i, :], pred2[i, :])
            if labels[i]:
                # if is positive pair, minimize distance
                loss.append(1 - d)
            else:
                # if is negative pair, minimize (margin - distance) if distance < margin
                loss.append(torch.clip((d - self.margin), min=0.))
        return torch.mean(torch.stack(loss))
