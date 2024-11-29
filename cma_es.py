import numpy as np
import logging
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.ga import GA
from pathlib import Path
from torch.nn import CrossEntropyLoss as CEL
from torchmetrics import Accuracy
import torch

from cifar100_ensemble_selection_weights import Evaluator
from models import load_cifar10_models, load_cifar100_models


class SimpleProblem(ElementwiseProblem):
    def __init__(
        self,
        model_lib,
        evaluator,
        pipeline,
        augment_mask=True,
        use_both_lighting=False,
    ):
        self.evaluator = evaluator
        self.model_lib = model_lib
        self.N_models = len(model_lib)
        self.augment_mask = augment_mask
        self.use_both_lighting = use_both_lighting

        super().__init__(n_var=self.N_models, n_obj=1, n_ieq_constr=1, xl=0.9, xu=1.1)

    def _evaluate(self, voting_weights, out, *args, **kwargs):
        ensemble = []
        for i, n in enumerate(voting_weights):
            if n:
                ensemble.append(self.model_lib[i])
        norm_weights = [float(w)/sum(voting_weights) for w in voting_weights]

        score = self.evaluator.run(ensemble, norm_weights, "validation")

        print(voting_weights, score)

        out["F"] = score
        out["G"] = -out["F"]


def optimize_CMAES(
    optimization_cfg,
    model_lib,
    nr_classes,
    scoring_function,
    penalty,
    seed=0,
    augment_mask=True,
    use_pseudo_label=False,
    use_both_lighting=False,
    optimise_order=False
):
    evaluator = Evaluator(nr_classes, scoring_function, penalty, use_pseudo_label)

    algorithm = GA(
        pop_size=optimization_cfg["n_pop"],
        eliminate_duplicates=True)

    problem = SimpleProblem(
        model_lib, evaluator, pipeline=None
    )

    result = minimize(
        problem,
        algorithm,
        ("n_gen", optimization_cfg["n_gen"]),
        seed=seed,
        verbose=True,
    )

    return result.X, result.F



model_lib = load_cifar100_models()
num_class = 100
scoring_function = CEL()
penalty = 0

best_candidate, score = optimize_CMAES(optimization_cfg={"n_pop":50, "n_gen":20}, model_lib=model_lib, nr_classes=num_class, scoring_function=scoring_function, penalty=penalty)
print(best_candidate, score)

ensemble = []
for i, n in enumerate(best_candidate):
    if n:
        ensemble.append(model_lib[i])

evaluator = Evaluator(num_class, scoring_function, penalty, False)
scores = []
accuracy = Accuracy(task='multiclass', num_classes=num_class)
norm_weights = [float(w)/sum(best_candidate) for w in best_candidate]
for images, lbl in evaluator.testset:
    all_outputs = []
    for i, m in enumerate(ensemble):
        model_pred = m(images).detach().numpy()
        model_weights = np.empty_like(model_pred)
        model_weights.fill(norm_weights[i])
        all_outputs.append(model_pred * model_weights)
    output = sum(all_outputs)
    scores.append(accuracy(target=torch.Tensor(lbl), preds=torch.Tensor(output)))
score = np.mean(scores)
print(f"Best accuracy on testset without penalty: {score}")
