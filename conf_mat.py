import torch

"""
Class implementing Confusion Matrix and calculating MIoU based on it.
Taken from https://github.com/pytorch/vision/blob/main/references/segmentation/utils.py
"""
class ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.inference_mode():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        acc = torch.nan_to_num(acc, 1) # nan -> 1 in case the class is not present in the sample
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        iu = torch.nan_to_num(iu, 1) # nan -> 1 in case the class is not present in the sample
        return acc_global, acc, iu

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return ("global correct: {:.1f}\naverage row correct: {}\nIoU: {}\nmean IoU: {:.1f}").format(
            acc_global.item() * 100,
            [f"{i:.1f}" for i in (acc * 100).tolist()],
            [f"{i:.1f}" for i in (iu * 100).tolist()],
            iu.mean().item() * 100,
        )
