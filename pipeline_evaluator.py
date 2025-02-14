import numpy as np
import torch
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import logging
from transformations import adjust_brightness, adjust_contrast

class Evaluator:
    def __init__(
        self, dataset, scoring_fn, penalty, pseudo_labels=False
    ):
        self.dataset = dataset
        self.score_fn = scoring_fn
        self.penalty = penalty
        self.use_pseudo_label = pseudo_labels

    def run(self, models, pipeline=None):
        scores = []
        counter = 0
        for images, labels in self.dataset:
            counter += 1
#            print(images.shape)
            images = adjust_brightness(images, 0.6)
#            images = torch.Tensor(images)
#            images = adjust_contrast(images, 0.6)
#            images = torch.Tensor(images)
            if self.use_pseudo_label:
                lbl = labels #pseudo_labels
            else:
                lbl = labels
            if pipeline is not None:
                images, lbl = pipeline(images, lbl)
                images = torch.Tensor(images)
                lbl = torch.Tensor(lbl)

            if len(models) > 1:
                all_outputs = []
                for i, m in enumerate(models):
                    model_pred = m[0](images)['out'].detach().numpy()
                    model_weights = np.empty_like(model_pred)
                    model_weights.fill(m[1])
                    all_outputs.append(model_pred * model_weights)
                output = sum(all_outputs)
            else:
                output = models[0][0](images)['out']

            batch_size = 30
            loss = self.score_fn(torch.Tensor(output), torch.Tensor(lbl).to(torch.long).squeeze(), ignore_index=255)
            scores.append(loss.item())

        return scores
