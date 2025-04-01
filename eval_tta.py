import logging
from pathlib import Path

import cv2
import hydra
import openpyxl
import numpy as np
from omegaconf import DictConfig
from torch import nn
import torchvision.transforms.functional as F
import torch
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from pipeline_evaluator_individual import maybe_rescale_prediction
from pipelineGA_individual import optimize_Sequential_ES as individual_Seq_ES
from pipelineGA_general import optimize_Sequential_ES as general_Seq_ES
from new_pipeline_evolution import optimize_Sequential_ES
from data import load_PascalVOC_pipeline, CIFARData
import transformations
from models import load_pascal_weighted_models, load_pascal_models, load_cifar100_models, load_best_cifar100, load_best_pascal, load_cifar_ensemble
from conf_mat import ConfusionMatrix

@hydra.main(version_base=None, config_path=".", config_name="tta")
def run(cfg: DictConfig):
    task = 'CIFAR'
    # Load a model
    if task == 'CIFAR':
        model_lib = load_best_cifar100()
        model_lib = [[m,1] for m in model_lib]
    else:
        model_lib = load_best_pascal()
        model_lib = [[m,1] for m in model_lib]
    logging.info(f"Number of models loaded: {len(model_lib)}")

    if task == 'CIFAR':
        train_samples, test_samples = CIFARData(100).test_dataloader()
        train_samples = DataLoader(
                train_samples,
                batch_size=150,
                num_workers=4,
                drop_last=True,
                pin_memory=True,
                shuffle=False,
            )
    else:
         train_samples, test_samples = load_PascalVOC_pipeline()

    # Optimize function
    scoring_fn = nn.functional.cross_entropy
    pipeline_GA_type = 'vanilla'
    if pipeline_GA_type == 'vanilla':
        optimiser = optimize_Sequential_ES
    elif pipeline_GA_type == 'general':
        optimiser = general_Seq_ES
    elif pipeline_GA_type == 'individual':
        optimiser = individual_Seq_ES
    else:
        optimiser = None

    for i in [12345, 52981, 80462]:

        # Optimization method
        optimized_pipeline, model_idx = optimiser(
            [cfg.augs, cfg.augs] if cfg.use_both_lighting else [cfg.augs],
            cfg.resize_cfg,
            model_lib,
            train_samples,
            scoring_fn,
            seed=i,
            augment_mask=cfg.augment_mask,
            use_both_lighting=cfg.use_both_lighting,
            use_pseudo_label=False,
#            optimise_order=True, #cfg.optimise_order,
#            optimise_ensemble=True,
        )

        if model_idx == -1:
            logging.info("Optimized pipeline statistics\n" f"{optimized_pipeline}")
        else:
            logging.info(
                "Optimized pipeline statistics\n"
                f"selected model: {model_idx}\n"
                f"{optimized_pipeline}"
            )
        
        # for when using an ensemble
        if task == 'CIFAR':
            scores = []
            accuracy = Accuracy(task='multiclass', num_classes=100)
        else:
            confmat = ConfusionMatrix(21)
        total_w = sum([tup[1] for tup in model_lib])
        norm_weights = [tup[1]/total_w for tup in model_lib]
        for images, lbl in test_samples:
            all_outputs = []
            for i, s in enumerate(model_lib):
                adjusted_images = transformations.adjust_brightness(images, 0.5)
                if pipeline_GA_type == 'individual':
                    piped_images, piped_lbls = optimized_pipeline[i](adjusted_images, lbl)
                else:
                    piped_images, piped_lbls = optimized_pipeline(adjusted_images, lbl)
                model_pred = s[0](torch.Tensor(piped_images))
                if task == 'CIFAR':
                    model_pred = model_pred.detach().numpy()
                else:
                    model_pred = model_pred['out'].detach().numpy()
                model_weights = np.empty_like(model_pred)
                model_weights.fill(norm_weights[i])
                weighted_pred = model_pred * model_weights
                all_outputs.append(maybe_rescale_prediction(weighted_pred, lbl.shape))
            output = sum(all_outputs)
            output, piped_lbls = torch.Tensor(output), torch.Tensor(piped_lbls)
            if task == 'CIFAR':
                scores.append(accuracy(target=torch.Tensor(lbl), preds=torch.Tensor(output)))
            else:
                confmat.update(lbl.flatten(), output.argmax(1).flatten())
        print("Best performance on testset without penalty:")
        if task == 'CIFAR':
            print(np.mean(scores))
        else:
            print(confmat)

if __name__ == "__main__":
    run()
