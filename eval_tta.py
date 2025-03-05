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

from pipeline_evaluator_individual import maybe_rescale_prediction
from pipelineGA_individual import optimize_Sequential_ES as individual_Seq_ES
from pipelineGA_general import optimize_Sequential_ES as general_Seq_ES
from new_pipeline_evolution import optimize_Sequential_ES
from data import load_PascalVOC_pipeline
import transformations
from models import load_pascal_weighted_models, load_pascal_models
from conf_mat import ConfusionMatrix

@hydra.main(version_base=None, config_path=".", config_name="tta")
def run(cfg: DictConfig):
    # Load a model
    model_lib = load_pascal_weighted_models()
    logging.info(f"Number of models loaded: {len(model_lib)}")

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

    for i in range(cfg.n_seeds):

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

        # for when using only one model
#        confmat = ConfusionMatrix(21)
#        for images, lbl in test_samples:
#            adjusted_images = adjust_brightness(images, 0.6)
#            piped_images, piped_lbls = optimized_pipeline(adjusted_images, lbl)
#            model_pred = model_lib[0](torch.Tensor(piped_images))
#            model_pred = model_pred['out'].detach().numpy()
#            print(model_pred.shape, piped_lbl.shape)
#            confmat.update(torch.Tensor(piped_lbls.flatten()), torch.Tensor(model_pred.argmax(1).flatten()))
#        print("Best performance on testset without penalty:")
#        print(confmat)

        # for when using an ensemble
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
                model_pred = model_pred['out'].detach().numpy()
                model_weights = np.empty_like(model_pred)
                model_weights.fill(norm_weights[i])
                weighted_pred = model_pred * model_weights
                all_outputs.append(maybe_rescale_prediction(weighted_pred, lbl.shape))
            output = sum(all_outputs)
            output, piped_lbls = torch.Tensor(output), torch.Tensor(piped_lbls)
            confmat.update(lbl.flatten(), output.argmax(1).flatten())
        print("Best performance on testset without penalty:")
        print(confmat)

if __name__ == "__main__":
    run()
