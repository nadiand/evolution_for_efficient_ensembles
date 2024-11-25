import logging
import os
import torch
import numpy as np
import time

from data import CIFAR10Data
from cifar10_models.densenet import densenet121, densenet169, densenet161
from cifar10_models.resnet import resnet18, resnet34
from cifar10_models.googlenet import googlenet
from cifar10_models.mobilenetv2 import mobilenet_v2
from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn
from torchmetrics import Accuracy
from torch.nn import CrossEntropyLoss as CEL


# Candidate class that contains the parameters and order
class Candidate:
    def __init__(self, voting_weights):
        self.voting_weights = voting_weights
        self.fitness = None

    # Evaluates the candidates parameters in the given order
    def __call__(self):
        return self.voting_weights

    def set_fitness(self, fitness):
        self.fitness = fitness

class SimpleProblem:
    def __init__(
        self, model_lib, evaluator, pipeline, augment_mask=True, use_both_lighting=False
    ):
        self.model_lib = model_lib
        self.evaluator = evaluator
        self.pipeline = pipeline
        self.augment_mask = augment_mask
        self.use_both_lighting = use_both_lighting

    def __call__(self, voting_weights, eval_type):
        ensemble = []
        for i, n in enumerate(voting_weights):
            if n:
                ensemble.append(self.model_lib[i])
        norm_weights = [float(w)/sum(voting_weights) for w in voting_weights]
        score = self.evaluator.run(ensemble, norm_weights, eval_type)
        return score

class Evaluator:
    def __init__(
        self, scoring_fn, penalty, pseudo_labels=False
    ):
        self.dataset, self.testset = CIFAR10Data().test_dataloader()
        self.score_fn = scoring_fn
        self.penalty = penalty

    def run(self, models, weights, eval_type, pipeline=None):
        penalty = np.count_nonzero(weights)*self.penalty
        scores = []
        if eval_type == 'validation':
            dataset = self.dataset
        else:
            dataset = self.testset
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

            scores.append(self.score_fn(torch.Tensor(output), torch.Tensor(lbl)))

        score = np.mean(scores) + penalty
        return score

def initialize_population(population_size, N_models, threshold, history):
    population = []
    index = 0
    while len(population) < population_size:
        sequence = np.random.uniform(0.0, high=1.0, size=N_models)
        sequence = [s if s>threshold else 0 for s in sequence]
        if not np.all(sequence==history, axis=2).any():
            history[0, index] = sequence
            population.append(Candidate(sequence))
            index += 1
    return population, history

def mutate(ensemble, mutation_type, threshold, generation):
    N_models = len(ensemble)
    if mutation_type == "all":
        sigma = 0.2 #- generation*0.01
        factor = np.random.uniform(-sigma, sigma, size=N_models)
        ensemble += factor
        ensemble = [w if w>threshold else 0 for w in ensemble]
    elif mutation_type == "one":
        modified_model = np.random.randint(0, N_models)
        new_weight = np.random.uniform()
        ensemble[modified_model] = new_weight if new_weight > threshold else 0

    return ensemble

def reproduce_uniform(population, history, g, population_size, N_models, threshold):
    new_population = []
    index = 0
    fitnesses = np.array([1/p.fitness for p in population])
    prob = fitnesses / np.sum(fitnesses)

    while index < population_size:
        parents = np.random.choice(population, p=prob, size=2, replace=False)
        genes_parent1 = np.random.uniform(0.0, 1.0, size=N_models) > 0.5
        child = []
        for i, gene in enumerate(genes_parent1):
            child.append(parents[0].voting_weights[i] if gene else parents[1].voting_weights[i])

        if np.random.uniform() < 0.25:
            child = mutate(child, "all", threshold, g-1)

        if not (np.all(child==history, axis=2).any()):
            history[g, index] = child
            new_population.append(Candidate(child))
            index += 1

    return new_population, history


def reproduce(population, history, g, population_size, N_models, threshold):
    new_population = []
    index = 0
    fitnesses = np.array([1/p.fitness for p in population])
    prob = fitnesses / np.sum(fitnesses)

    while index < population_size:
        parents = np.random.choice(population, p=prob, size=2, replace=False)
        cX_point = np.random.randint(1, N_models-1)

        child1 = np.zeros(N_models, dtype=float)
        child1[:cX_point] = parents[0].voting_weights[:cX_point]
        child1[cX_point:] = parents[1].voting_weights[cX_point:]

        if True: #np.random.uniform() < 0.75:
            child1 = mutate(child1, "all", threshold, g-1)

        child2 = np.zeros(N_models, dtype=float)
        child2[:cX_point] = parents[1].voting_weights[:cX_point]
        child2[cX_point:] = parents[0].voting_weights[cX_point:]

        if True: #np.random.uniform() < 0.75:
            child2 = mutate(child2, "all", threshold, g-1)

        if not (np.all(child1==history, axis=2).any()):
            history[g, index] = child1
            new_population.append(Candidate(child1))
            index += 1

        if index < population_size:
            if not (np.all(child2==history, axis=2).any()):
                history[g, index] = child2
                new_population.append(Candidate(child2))
                index += 1
    return new_population, history


def evaluate_population(population, problem, eval_type):
    for candidate in population:
        fitness = problem(candidate(), eval_type)
        candidate.set_fitness(fitness)
    return population

def select_with_elitism(parents, offspring, population_size):
    parents.sort(key=lambda x: x.fitness)
    new_population = parents[:int(population_size*0.2)] # 20% of the old population, the elites, get to be part of new population
    offspring.sort(key=lambda x: x.fitness)
    i = 0
    while len(new_population) < population_size:
        new_population.append(offspring[i])
        i += 1
    new_population.sort(key=lambda x: x.fitness)
    return new_population[0], new_population


def select(parents, offspring, population_size):
    selection_pool = parents + offspring
    selection_pool.sort(key=lambda x: x.fitness)

    return selection_pool[0], selection_pool[:population_size]

def select_ensemble(model_lib, scoring_fn, seed=0, pipeline = None,
        use_both_lighting=False, use_pseudo_label=False, augment_mask=True):

    penalty = 0.05
    evaluator = Evaluator(scoring_fn, penalty, use_pseudo_label)
    problem = SimpleProblem(model_lib, evaluator, pipeline, augment_mask, use_both_lighting)

    n_gen = 20
    population_size = 50
    N_models = len(model_lib)
    threshold = 0.1

    np.random.seed(seed=seed+2)

    history = np.zeros((n_gen+1, population_size, N_models))
    population, history = initialize_population(population_size, N_models, threshold, history)
    population = evaluate_population(population, problem, 'validation')
    print("Init pop:")
    for p in population:
        print(p.voting_weights, p.fitness)
    time_per_epoch = 0
    for g in range(n_gen):
        start_epoch = time.time()
        offspring, history = reproduce_uniform(population, history, g+1, population_size, N_models, threshold)
        offspring = evaluate_population(offspring, problem, 'validation')
        best_candidate, population = select_with_elitism(population, offspring, population_size)
        end_epoch = time.time()
        time_per_epoch += (end_epoch - start_epoch)
        print("Current population:")
        for p in population:
            print(p.voting_weights, p.fitness)
        # logging.info(f"Generation: {g+1}, best fitness: {best_candidate.fitness}, ensemble: {best_candidate.voting_weights}")
        print((f"Generation: {g+1}, best fitness: {best_candidate.fitness}, ensemble: {best_candidate.voting_weights}"))
        print("-"*80)

    best_candidate = evaluate_population([best_candidate], problem, 'test')[0]
    fitness_no_penalty = best_candidate.fitness - penalty*np.count_nonzero(best_candidate.voting_weights)
    print(f"Best fitness on testset without penalty: {fitness_no_penalty}")

    ensemble = []
    for i, n in enumerate(best_candidate.voting_weights):
        if n:
            ensemble.append(problem.model_lib[i])

    scores = []
    accuracy = Accuracy(task='multiclass', num_classes=10)
    for images, lbl in evaluator.testset:
        all_outputs = []
        for i, m in enumerate(ensemble):
            model_pred = m(images).detach().numpy()
            model_weights = np.empty_like(model_pred)
            model_weights.fill(best_candidate.voting_weights[i])
            all_outputs.append(model_pred * model_weights)
        output = sum(all_outputs)
        scores.append(accuracy(target=torch.Tensor(lbl), preds=torch.Tensor(output)))
    score = np.mean(scores)
    print(f"Best accuracy on testset without penalty: {score}")

    print(f"Total time it took to do {n_gen} epochs: {time_per_epoch}")
    print(f"Time it took to do one epoch: {time_per_epoch/n_gen}")

    return best_candidate.voting_weights


def load_models():
    classifiers = []

    densenet_model = densenet121()
    state_dict = os.path.join("cifar10_models", "state_dicts", "densenet121" + ".pt")
    densenet_model.load_state_dict(torch.load(state_dict))
    densenet_model.eval()
    classifiers.append(densenet_model)

    densenet_model2 = densenet169()
    state_dict = os.path.join("cifar10_models", "state_dicts", "densenet169" + ".pt")
    densenet_model2.load_state_dict(torch.load(state_dict))
    densenet_model2.eval()
    classifiers.append(densenet_model2)

    resnet_model = resnet18()
    state_dict = os.path.join("cifar10_models", "state_dicts", "resnet18" + ".pt")
    resnet_model.load_state_dict(torch.load(state_dict))
    resnet_model.eval()
    classifiers.append(resnet_model)

    resnet_model2 = resnet34()
    state_dict = os.path.join("cifar10_models", "state_dicts", "resnet34" + ".pt")
    resnet_model2.load_state_dict(torch.load(state_dict))
    resnet_model2.eval()
    classifiers.append(resnet_model2)

    googlenet_model = googlenet()
    state_dict = os.path.join("cifar10_models", "state_dicts", "googlenet" + ".pt")
    googlenet_model.load_state_dict(torch.load(state_dict))
    googlenet_model.eval()
    classifiers.append(googlenet_model)

    mobilenet_v2_model = mobilenet_v2()
    state_dict = os.path.join("cifar10_models", "state_dicts", "mobilenet_v2" + ".pt")
    mobilenet_v2_model.load_state_dict(torch.load(state_dict))
    mobilenet_v2_model.eval()
    classifiers.append(mobilenet_v2_model)

    densenet_model3 = densenet161()
    state_dict = os.path.join("cifar10_models", "state_dicts", "densenet161" + ".pt")
    densenet_model3.load_state_dict(torch.load(state_dict))
    densenet_model3.eval()
    classifiers.append(densenet_model3)

    vgg_model = vgg11_bn()
    state_dict = os.path.join("cifar10_models", "state_dicts", "vgg11_bn" + ".pt")
    vgg_model.load_state_dict(torch.load(state_dict))
    vgg_model.eval()
    classifiers.append(vgg_model)

    vgg_model2 = vgg13_bn()
    state_dict = os.path.join("cifar10_models", "state_dicts", "vgg13_bn" + ".pt")
    vgg_model2.load_state_dict(torch.load(state_dict))
    vgg_model2.eval()
    classifiers.append(vgg_model2)

    vgg_model3 = vgg16_bn()
    state_dict = os.path.join("cifar10_models", "state_dicts", "vgg16_bn" + ".pt")
    vgg_model3.load_state_dict(torch.load(state_dict))
    vgg_model3.eval()
    classifiers.append(vgg_model3)

    print('models loaded')
    return classifiers

    # Evaluating the individual models to have a baseline of performance
    accuracy = Accuracy(task='multiclass', num_classes=10)
    loss_fn = CEL()
    val_dataset, test_dataset = CIFAR10Data().test_dataloader()
    for i, c in enumerate(classifiers):
        scores = []
        for images, lbl in test_dataset:
            model_pred = c(images)
            loss = loss_fn(model_pred, lbl).detach().numpy()
            scores.append(loss)
        print(f'model {i} has loss {np.mean(scores)}')

    scores = []
    accuracies = []
    for images, lbl in test_dataset:
        all_outputs = []
        for i, m in enumerate(classifiers):
            model_pred = m(images).detach().numpy()
            all_outputs.append(model_pred)
        output = np.mean(all_outputs, axis=0)
        scores.append(loss_fn(torch.Tensor(output), torch.Tensor(lbl)))
        accuracies.append(accuracy(target=torch.Tensor(lbl), preds=torch.Tensor(output)))
    print(f'baseline ensemble loss is {np.mean(scores)}, and accuracy is {np.mean(accuracies)}')

    return classifiers


if __name__ == "__main__":

    best_voting_weights = select_ensemble(model_lib=load_models(), scoring_fn=CEL(), seed=0, pipeline = None,
        use_both_lighting=False, use_pseudo_label=False, augment_mask=True)
    print(best_voting_weights)
