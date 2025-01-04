import numpy as np
import torch
from torch.utils.data import DataLoader
from data import CIFARData, load_PascalVOC
import torchvision.transforms.functional as F


class Evaluator:
    def __init__(
        self, nr_classes, scoring_fn, penalty, pseudo_labels=False
    ):
        self.val_dataset, self.test_loader = CIFARData(nr_classes).test_dataloader()
        self.score_fn = scoring_fn
        self.penalty = penalty

    def run(self, models, weights, eval_type, sampler, pipeline=None):
        penalty = len(weights)*self.penalty
        scores = []

        if eval_type == 'validation':
            data = self.val_dataset
            val_dataloader = DataLoader(
                data,
                batch_size=30,
                num_workers=4,
                drop_last=True,
                pin_memory=True,
                sampler=sampler,
                shuffle=False,
            )
            dataset = val_dataloader
        else:
            dataset = self.test_loader

        for images, lbl in dataset:
            if pipeline is not None:
                images, lbl = pipeline(images, lbl)
            if len(models) > 1:
                all_outputs = []
                for i, m in enumerate(models):
                    model_pred = m(images).detach().numpy()
                    model_weights = np.empty_like(model_pred)
                    model_weights.fill(weights[i])
                    all_outputs.append(model_pred * model_weights)
                output = sum(all_outputs)
            else:
                output = models[0](images)

            scores.append(self.score_fn(torch.Tensor(output), torch.Tensor(lbl)).detach().numpy())

        score = np.mean(scores) + penalty
        return score


class EvaluatorSegmentation:
    def __init__(
        self, nr_classes, scoring_fn, penalty, pseudo_labels=False
    ):
        self.val_dataset, self.test_loader = load_PascalVOC()
        self.score_fn = scoring_fn
        self.penalty = penalty
        self.num_classes = nr_classes

    def run(self, models, weights, eval_type, sampler, pipeline=None):
        penalty = len(weights)*self.penalty
        scores = []

        if eval_type == 'validation':
            batch_size = 30
            data = self.val_dataset
            val_dataloader = DataLoader(
                data,
                batch_size=batch_size,
                num_workers=4,
                drop_last=True,
                pin_memory=True,
                sampler=sampler,
                shuffle=False,
            )
            dataset = val_dataloader
        else:
            batch_size = 1
            dataset = self.test_loader

        for images, lbl in dataset:
            adjusted_images = F.adjust_brightness(images, brightness_factor=0.8)
            if len(models) > 1:
                all_outputs = []
                for i, m in enumerate(models):
                    model_pred = m(adjusted_images)
                    model_pred = model_pred['out'].detach().numpy()
                    model_weights = np.empty_like(model_pred)
                    model_weights.fill(weights[i])
                    all_outputs.append(model_pred * model_weights)
                output = sum(all_outputs)
            else:
                output = models[0](adjusted_images)['out']

            loss = self.score_fn(torch.Tensor(output), torch.Tensor(lbl).to(torch.long).reshape((batch_size,520,520)), ignore_index=255)
            scores.append(loss.item())

        score = np.mean(scores) + penalty
        return score
