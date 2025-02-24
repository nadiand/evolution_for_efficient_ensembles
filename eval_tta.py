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

from pipeline_evolution import optimize_Sequential_ES
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

    for i in range(cfg.n_seeds):

        # Optimization method
        optimized_pipeline, model_idx = optimize_Sequential_ES(
            [cfg.augs, cfg.augs] if cfg.use_both_lighting else [cfg.augs],
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
        norm_weights = [1]
#        norm_weights = [float(w)/sum([model_lib[0][1], model_lib[1][1], model_lib[2][1], model_lib[3][1]]) for w in [model_lib[0][1], model_lib[1][1], model_lib[2][1], model_lib[3][1]]] # TODO make this nonhardcoded
        for images, lbl in test_samples:
            all_outputs = []
            for i, s in enumerate(model_lib):
                adjusted_images = transformations.adjust_brightness(images, 0.6)
                piped_images, piped_lbls = optimized_pipeline(adjusted_images, lbl)
                model_pred = s[0](torch.Tensor(piped_images))
                model_pred = model_pred['out'].detach().numpy()
                model_weights = np.empty_like(model_pred)
                model_weights.fill(norm_weights[i])
                all_outputs.append(model_pred * model_weights)
            output = sum(all_outputs)
            output, piped_lbls = torch.Tensor(output), torch.Tensor(piped_lbls)
            confmat.update(piped_lbls.flatten(), output.argmax(1).flatten())
        print("Best performance on testset without penalty:")
        print(confmat)

if __name__ == "__main__":
    run()
