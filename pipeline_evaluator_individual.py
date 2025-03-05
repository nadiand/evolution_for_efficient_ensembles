import numpy as np
import torch
import torchvision.transforms.functional as F
from torch.nn import functional as func
from torch.utils.data import DataLoader
import logging
import transformations

def maybe_rescale_prediction(pred, tgt_shape, mode='bilinear'):
    if len(pred.shape) == 4 and pred.shape[2:] != tgt_shape[2:]:
        pred = func.interpolate(torch.Tensor(pred), tgt_shape[2:], mode=mode, align_corners=False)
    return pred


class Evaluator:
    def __init__(
        self, dataset, scoring_fn, penalty, pseudo_labels=False
    ):
        self.dataset = dataset
        self.score_fn = scoring_fn
        self.penalty = penalty
        self.use_pseudo_label = pseudo_labels

    def run(self, models, pipelines=None):
        scores = []
        counter = 0
        for images, labels in self.dataset:
            counter += 1
            images = transformations.adjust_brightness(images, 0.5)
            if self.use_pseudo_label:
                lbl = labels #pseudo_labels
            else:
                lbl = labels

            if len(models) > 1:
                all_outputs = []
                for i, m in enumerate(models):
                    if pipelines[i] is not None:
                        images, lbl_r = pipelines[i](images, lbl)

                    model_pred = m[0](images)['out'].detach().numpy()
                    model_weights = np.empty_like(model_pred)
                    model_weights.fill(m[1])
                    weighted_pred = model_pred * model_weights
                    all_outputs.append(maybe_rescale_prediction(weighted_pred, lbl.shape))
                output = sum(all_outputs)
            else:
                if pipelines[0] is not None:
                    images, lbl_r = pipelines[0](images, lbl)

                output = models[0][0](images)['out']
                output = maybe_rescale_prediction(output, lbl.shape)

            batch_size = 30
            loss = self.score_fn(torch.Tensor(output), torch.Tensor(lbl).to(torch.long).squeeze(), ignore_index=255)
            scores.append(loss.item())

        return scores
