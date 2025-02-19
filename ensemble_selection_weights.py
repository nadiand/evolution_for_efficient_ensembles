import logging
import os
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.utils.data import SubsetRandomSampler, DataLoader
from torch import nn

from models import load_cifar10_models, load_cifar100_models, load_pascal_models, load_pascal_preds
from data import CIFARData, load_PascalVOC
from conf_mat import ConfusionMatrix
from evaluator import Evaluator, EvaluatorSegmentation, EvaluatorPredictions
from candidate import Candidate, SimpleProblem
from diversity_metrics import pierson_correlation
from torchmetrics import Accuracy
from torch.nn import CrossEntropyLoss as CEL
import torchvision.transforms.functional as F


def initialize_population(population_size, N_models, threshold, history, g):
    population = []
    index = 0
    while len(population) < population_size:
        # Completely random weights in initial pop
        # sequence = np.random.uniform(0.0, high=1.0, size=N_models)
        # Weights close to 1, i.e. all models get almost equal voting power
        sequence = np.random.normal(loc=1.0, scale=0.2, size=N_models)
        sequence = [s if s>threshold else 0 for s in sequence]
        if not np.all(sequence==history, axis=2).any():
            history[0, index] = sequence
            population.append(Candidate(sequence, g))
            index += 1
    return population, history

def mutate(ensemble, mutation_type, threshold, generation):
    N_models = len(ensemble)
    if mutation_type == "all":
        sigma = 0.2 - generation*0.01
        factor = np.random.uniform(-sigma, sigma, size=N_models)
        ensemble += factor
        ensemble = [w if w>threshold else 0 for w in ensemble]
    elif mutation_type == "one":
        modified_model = np.random.randint(0, N_models)
        new_weight = np.random.uniform()
        ensemble[modified_model] = new_weight if new_weight > threshold else 0

    return ensemble


def mutate_population(population, history, g, population_size, threshold):
    new_population = []
    index = 0
    fitnesses = np.array([1/p.fitness for p in population])
    prob = fitnesses / np.sum(fitnesses)

    while index < population_size:
        individual = np.random.choice(population, p=prob, size=1)[0]
        child = mutate(individual.voting_weights, "all", threshold, g-1)
        if not (np.all(child==history, axis=2).any()):
            history[g, index] = child
            new_population.append(Candidate(child, g))
            index += 1

    return new_population, history
    

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

        if True: #np.random.uniform() < 0.5:
            child = mutate(child, "all", threshold, g-1)

        if not (np.all(child==history, axis=2).any()):
            history[g, index] = child
            new_population.append(Candidate(child, g))
            index += 1

    return new_population, history


def reproduce(population, history, g, population_size, N_models, threshold):
    new_population = []
    index = 0
    fitnesses = np.array([1/p.fitness for p in population])
    prob = fitnesses / np.sum(fitnesses)

    while index < population_size:
        # Just picking two random models using distribution
        parents = np.random.choice(population, p=prob, size=2, replace=False)

        # Doing tournament selection
#        tournament_pool1 = np.random.choice(population, size=int(population_size*0.1), replace=False)
#        tournament_pool2 = np.random.choice(population, size=int(population_size*0.1), replace=False)
#        fitnesses1 = np.array([p.fitness for p in tournament_pool1])
#        fitnesses2 = np.array([p.fitness for p in tournament_pool2])
#        ages2 = np.array([p.generation for p in tournament_pool2])
#        print(fitnesses1, ages2)
#        winner1 = tournament_pool1[np.argmin(fitnesses1)]
#        winner2 = tournament_pool2[np.argmin(fitnesses2)]
#        parents = [winner1, winner2]

        cX_point = np.random.randint(1, N_models-1)

        child1 = np.zeros(N_models, dtype=float)
        child1[:cX_point] = parents[0].voting_weights[:cX_point]
        child1[cX_point:] = parents[1].voting_weights[cX_point:]

        if np.random.uniform() < 0.25:
            child1 = mutate(child1, "all", threshold, g-1)

        child2 = np.zeros(N_models, dtype=float)
        child2[:cX_point] = parents[1].voting_weights[:cX_point]
        child2[cX_point:] = parents[0].voting_weights[cX_point:]

        if np.random.uniform() < 0.25:
            child2 = mutate(child2, "all", threshold, g-1)

        if not (np.all(child1==history, axis=2).any()):
            history[g, index] = child1
            new_population.append(Candidate(child1, g))
            index += 1

        if index < population_size:
            if not (np.all(child2==history, axis=2).any()):
                history[g, index] = child2
                new_population.append(Candidate(child2, g))
                index += 1
    return new_population, history


def evaluate_population(population, problem, eval_type, gen, load_preds=False):
    if gen == 0:
        nr_samples = 50
    else:
        nr_samples = 30
    indices = np.random.choice(range(0, 50), size=nr_samples, replace=False)
    sampler = SubsetRandomSampler(indices=indices)
    for candidate in population:
        if load_preds:
            fitness = problem(candidate(), eval_type, indices)
        else:
            fitness = problem(candidate(), eval_type, sampler)
        candidate.set_fitness(fitness)
    return population

def select_with_elitism(population, population_size, curr_g):
    old, new = [], []
    for p in population:
        if p.generation == curr_g:
            new.append(p)
        elif p.generation < curr_g:
            old.append(p)
    old.sort(key=lambda x: x.fitness)
    new_population = old[:int(population_size*0.2)] # 20% of the old population, the elites, get to be part of new population
    new.sort(key=lambda x: x.fitness)
    i = 0
    while len(new_population) < population_size:
        new_population.append(new[i])
        i += 1
    new_population.sort(key=lambda x: x.fitness)
    return new_population[0], new_population


def select(selection_pool, population_size):
    selection_pool.sort(key=lambda x: x.fitness)

    return selection_pool[0], selection_pool[:population_size]


def select_ensemble(model_lib, nr_classes, scoring_fn, seed=0, pipeline = None,
        use_both_lighting=False, use_pseudo_label=False, augment_mask=True, load_preds=False):

    penalty = 0.01
    if load_preds:
        evaluator = EvaluatorPredictions(nr_classes, scoring_fn, penalty, use_pseudo_label)
    else:
        evaluator = EvaluatorSegmentation(nr_classes, scoring_fn, penalty, use_pseudo_label)
    problem = SimpleProblem(model_lib, evaluator, pipeline, augment_mask, use_both_lighting)

    n_gen = 15
    population_size = 30
    N_models = len(model_lib)
    threshold = 0.7

    np.random.seed(seed=seed+2)

    history = np.zeros((n_gen+1, population_size, N_models))
    population, history = initialize_population(population_size, N_models, threshold, history, 0)
    population = evaluate_population(population, problem, 'validation', n_gen, load_preds=load_preds)
    print("Init pop:")
    for p in population:
        print(p.voting_weights, p.fitness, p.generation)
    time_per_epoch = 0
    for g in range(n_gen):
        start_epoch = time.time()
        offspring, history = reproduce_uniform(population, history, g+1, population_size, N_models, threshold)
        candidate_population = population + offspring
        candidate_population = evaluate_population(candidate_population, problem, 'validation', n_gen-(g+1), load_preds=load_preds)
        best_candidate, population = select_with_elitism(candidate_population, population_size, g+1)
        end_epoch = time.time()
        time_per_epoch += (end_epoch - start_epoch)
        print((f"Generation: {g+1}, best fitness: {best_candidate.fitness}, ensemble: {best_candidate.voting_weights}, gen found: {best_candidate.generation}"))
        print("-"*80)

    print(f"Total time it took to do {n_gen} epochs: {time_per_epoch}")
    print(f"Time it took to do one epoch: {time_per_epoch/n_gen}")

    best_candidate = evaluate_population([best_candidate], problem, 'test', n_gen, load_preds=load_preds)[0]
    fitness_no_penalty = best_candidate.fitness - penalty*np.count_nonzero(best_candidate.voting_weights)
    print(f"Best fitness on testset without penalty: {fitness_no_penalty}")

    models = load_pascal_models()
    ensemble, weights = [], []
    for i, n in enumerate(best_candidate.voting_weights):
        if n:
            ensemble.append(models[i])
            weights.append(n)
    if nr_classes == 21:
        eval_best_segmentation(weights, ensemble, evaluator)
    else:
        eval_best_classification(weights, ensemble, nr_classes, evaluator)
    return best_candidate.voting_weights


def eval_best_segmentation(best_candidate, segmentors, evaluator):
    confmat = ConfusionMatrix(21)
    norm_weights = [float(w)/sum(best_candidate) for w in best_candidate]
    for images, lbl in evaluator.test_loader:
        adjusted_images = images*0.6
        adjusted_images[adjusted_images < -3] = -3
        adjusted_images[adjusted_images > 3] = 3
        
        all_outputs = []
        for i, s in enumerate(segmentors):
            model_pred = s(adjusted_images)
            model_weights = np.empty_like(model_pred)
            model_weights.fill(norm_weights[i])
            all_outputs.append(model_pred['out'].detach().numpy() * model_weights)
        output = sum(all_outputs)
        confmat.update(lbl.flatten(), output.argmax(1).flatten())
    print("Best performance on testset without penalty:")
    print(confmat)


def eval_best_classification(best_candidate, classifiers, nr_classes, evaluator):
    scores = []
    accuracy = Accuracy(task='multiclass', num_classes=nr_classes)
    norm_weights = [float(w)/sum(best_candidate) for w in best_candidate]
    for images, lbl in evaluator.test_loader:
        all_outputs = []
        for i, m in enumerate(classifiers):
            model_pred = m(images).detach().numpy()
            model_weights = np.empty_like(model_pred)
            model_weights.fill(norm_weights[i])
            all_outputs.append(model_pred * model_weights)
        output = sum(all_outputs)
        scores.append(accuracy(target=torch.Tensor(lbl), preds=torch.Tensor(output)))
    score = np.mean(scores)
    print(f"Best accuracy on testset without penalty: {score}")


def load_models(nr_classes, evaluate=False, load_preds=False):
    if nr_classes == 100:
        models = load_cifar100_models()
    elif nr_classes == 10:
        models = load_cifar10_models()
    elif nr_classes == 21:
        if load_preds:
            models = load_pascal_preds()
        else:
            models = load_pascal_models()

    if not evaluate:
        return models

    if nr_classes == 10 or nr_classes == 100:
        return evaluate_classification(nr_classes, models)
    else:
        return evaluate_segmentation(models)
        

def evaluate_segmentation(segmentors):
    _, test_dataset = load_PascalVOC()
    for i, s in enumerate(segmentors):
        confmat = ConfusionMatrix(21)
        s_preds = []
        for images, lbl in test_dataset:
            adjusted_images = images*0.6
            adjusted_images[adjusted_images < -3] = -3
            adjusted_images[adjusted_images > 3] = 3
            model_pred = s(adjusted_images)
            output = model_pred['out']
            s_preds.append(output.detach().numpy())
            confmat.update(lbl.flatten(), output.argmax(1).flatten())
        s_proba_arr = np.array(s_preds)
        print(f"stats for model {i}:")
        print(confmat)
        np.save(f"/dataB3/nadia_dobreva/model{i}_preds", s_proba_arr.flatten())
        torch.save(torch.Tensor(np.array(s_proba_arr)), f"/dataB3/nadia_dobreva/model{i}_tensor_preds.pt")

    confmat = ConfusionMatrix(21)
    for images, lbl in test_dataset:
        all_outputs = []
        for i, s in enumerate(segmentors):
            adjusted_images = images*0.6
            adjusted_images[adjusted_images < -3] = -3
            adjusted_images[adjusted_images > 3] = 3
            model_pred = s(adjusted_images)
            all_outputs.append(model_pred['out'].detach().numpy())
        output = np.mean(all_outputs, axis=0)
        confmat.update(lbl.flatten(), output.argmax(1).flatten())

    print("baseline ensemble stats:")
    print(confmat)
    return segmentors


def evaluate_classification(nr_classes, classifiers):
    # Evaluating the individual models to have a baseline of performance
    accuracy = Accuracy(task='multiclass', num_classes=nr_classes)
    loss_fn = CEL()
    _, test_dataset = CIFARData(nr_classes).test_dataloader()
    preds = []
    for i, c in enumerate(classifiers):
        scores, accs = [], []
        c_preds = []
        for images, lbl in test_dataset:
            model_pred = c(images)
            loss = loss_fn(model_pred, lbl).detach().numpy()
            scores.append(loss)
            output = np.argmax(model_pred.detach().numpy(), axis=1)
            accs.append(accuracy(target=torch.Tensor(lbl), preds=torch.Tensor(output)))
            c_preds.append(output)
        c_preds_arr = np.array(c_preds)
        preds.append(c_preds_arr.flatten())
        print(f'model {i} has loss {np.mean(scores)}, and accuracy {np.mean(accs)}')

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
    nr_classes = 100
    scoring_fn = nn.functional.cross_entropy
    load_preds = False

    best_voting_weights = select_ensemble(model_lib=load_models(nr_classes, load_preds=load_preds), nr_classes=nr_classes, scoring_fn=scoring_fn, seed=0, 
                                          pipeline=None, use_both_lighting=False, use_pseudo_label=False, augment_mask=True, load_preds=load_preds)
    print(best_voting_weights)
