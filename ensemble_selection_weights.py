import logging
import os
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.utils.data import SubsetRandomSampler, DataLoader

from models import load_cifar10_models, load_cifar100_models, load_pascal_models
from data import CIFARData, load_PascalVOC
from conf_mat import ConfusionMatrix
from diversity_metrics import pierson_correlation
from torchmetrics import Accuracy
from torch.nn import CrossEntropyLoss as CEL


# Candidate class that contains the parameters and order
class Candidate:
    def __init__(self, voting_weights, generation):
        self.voting_weights = voting_weights
        self.fitness = None
        self.generation = generation

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

    def __call__(self, voting_weights, eval_type, sampler):
        ensemble = []
        for i, n in enumerate(voting_weights):
            if n:
                ensemble.append(self.model_lib[i])
        norm_weights = [float(w)/sum(voting_weights) for w in voting_weights]
        score = self.evaluator.run(ensemble, norm_weights, eval_type, sampler)
        return score

class Evaluator:
    def __init__(
        self, nr_classes, scoring_fn, penalty, pseudo_labels=False
    ):
        if nr_classes == 21:
            self.val_dataset, self.test_loader = load_PascalVOC()
        else:
            self.val_dataset, self.test_loader = CIFARData(nr_classes).test_dataloader()
        self.score_fn = scoring_fn
        self.penalty = penalty

    def run(self, models, weights, eval_type, sampler, pipeline=None):
        penalty = np.count_nonzero(weights)*self.penalty
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

            scores.append(self.score_fn(torch.Tensor(output), torch.Tensor(lbl)))

        score = np.mean(scores) + penalty
        return score

def initialize_population(population_size, N_models, threshold, history, g):
    population = []
    index = 0
    while len(population) < population_size:
        # Completely random weights in initial pop
        sequence = np.random.uniform(0.0, high=1.0, size=N_models)
        # Weights close to 1, i.e. all models get almost equal voting power
        # sequence = np.random.normal(loc=1.0, scale=0.2, size=N_models)
        sequence = [s if s>threshold else 0 for s in sequence]
        if not np.all(sequence==history, axis=2).any():
            history[0, index] = sequence
            population.append(Candidate(sequence, g))
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
#        winner2 = tournament_pool2[np.argmin(ages2)]
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


def evaluate_population(population, problem, eval_type):
    indices = np.random.randint(0, 50, 30)
    sampler = SubsetRandomSampler(indices=indices)
    for candidate in population:
        fitness = problem(candidate(), eval_type, sampler)
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

def select_ensemble(model_lib, nr_classes, scoring_fn, seed=0, pipeline = None,
        use_both_lighting=False, use_pseudo_label=False, augment_mask=True):

    penalty = 0.05
    evaluator = Evaluator(nr_classes, scoring_fn, penalty, use_pseudo_label)
    problem = SimpleProblem(model_lib, evaluator, pipeline, augment_mask, use_both_lighting)

    n_gen = 20
    population_size = 50
    N_models = len(model_lib)
    threshold = 0.1

    np.random.seed(seed=seed+2)

    history = np.zeros((n_gen+1, population_size, N_models))
    population, history = initialize_population(population_size, N_models, threshold, history, 0)
    population = evaluate_population(population, problem, 'validation')
    print("Init pop:")
    for p in population:
        print(p.voting_weights, p.fitness, p.generation)
    time_per_epoch = 0
    for g in range(n_gen):
        start_epoch = time.time()
        offspring, history = reproduce(population, history, g+1, population_size, N_models, threshold)
        offspring = evaluate_population(offspring, problem, 'validation')
        best_candidate, population = select_with_elitism(population, offspring, population_size)
        end_epoch = time.time()
        time_per_epoch += (end_epoch - start_epoch)
        print("Current population:")
        for p in population:
            print(p.generation)
#            print(p.voting_weights, p.fitness, p.generation)
        # logging.info(f"Generation: {g+1}, best fitness: {best_candidate.fitness}, ensemble: {best_candidate.voting_weights}")
        print((f"Generation: {g+1}, best fitness: {best_candidate.fitness}, ensemble: {best_candidate.voting_weights}, gen found: {best_candidate.generation}"))
        print("-"*80)

    best_candidate = evaluate_population([best_candidate], problem, 'test')[0]
    fitness_no_penalty = best_candidate.fitness - penalty*np.count_nonzero(best_candidate.voting_weights)
    print(f"Best fitness on testset without penalty: {fitness_no_penalty}")

    ensemble = []
    for i, n in enumerate(best_candidate.voting_weights):
        if n:
            ensemble.append(problem.model_lib[i])

    scores = []
    accuracy = Accuracy(task='multiclass', num_classes=nr_classes)
    norm_weights = [float(w)/sum(best_candidate.voting_weights) for w in best_candidate.voting_weights]
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

    print(f"Total time it took to do {n_gen} epochs: {time_per_epoch}")
    print(f"Time it took to do one epoch: {time_per_epoch/n_gen}")

    return best_candidate.voting_weights


def load_models(nr_classes, evaluate=False):
    if nr_classes == 100:
        models = load_cifar100_models()
    elif nr_classes == 10:
        models = load_cifar10_models()
    elif nr_classes == 21:
        models = load_pascal_models()

    if not evaluate:
        return models

    if nr_classes == 10 or nr_classes == 100:
        return evaluate_classification(nr_classes, models)
    else:
        return evaluate_segmentation(models)
        

def evaluate_segmentation(segmentors):
    _, test_dataset = load_PascalVOC()
    all_proba_preds = []
    for i, s in enumerate(segmentors):
        confmat = ConfusionMatrix(21)
        s_preds = []
        for images, lbl in test_dataset:
            model_pred = s(images)
            output = model_pred['out']
            s_preds.append(output.detach().numpy())
            confmat.update(lbl.flatten(), output.argmax(1).flatten())
        s_proba_arr = np.array(s_preds)
        all_proba_preds.append(s_proba_arr.flatten())
        print(f"stats for model {i}:")
        print(confmat)

    piersons_dict = pierson_correlation(all_proba_preds)
    print("diversity of models according to pierson correlation coefficient:")
    print('pearsons coeff with pval<0.05 and abs(val) above 0.5')
    for k in piersons_dict.keys():
        if piersons_dict[k][1] < 0.05 and abs(piersons_dict[k][0]) > 0.5:
            print(k, piersons_dict[k])

    confmat = ConfusionMatrix(21)
    for images, lbl in test_dataset:
        all_outputs = []
        for i, s in enumerate(segmentors):
            model_pred = s(images)
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

    best_voting_weights = select_ensemble(model_lib=load_models(nr_classes), nr_classes=nr_classes, scoring_fn=CEL(), seed=0, pipeline = None,
        use_both_lighting=False, use_pseudo_label=False, augment_mask=True)
    print(best_voting_weights)
