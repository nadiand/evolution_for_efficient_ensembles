import numpy as np
import logging
from pipeline_evaluator_individual import Evaluator
from image_pipeline import ImagePipeline
import time
import torch


# Candidate class that contains the parameters and order
class Candidate:
    def __init__(self, params, order=None, factors=[1,1,1,1,1,1,1,1], model_idx=0):
        self.model_idx = model_idx # the model whose pipeline we're currently optimising
        self.params = params
        self.order = order
        self.factors = factors
        self.fitness = None

    # Evaluates the candidates parameters in the given order
    def __call__(self):
        return self.params, self.order, self.factors, self.model_idx

    def set_fitness(self, fitness):
        self.fitness = fitness

def sample_params(population, population_size, sigmas, resize_cfg, bounds, N_models, change_model_p, optimize_order, pipeline_to_optimise):
    new_population = []
    fitnesses = np.array([-p.fitness for p in population])
    rng = np.random.default_rng()
    prob = fitnesses / np.sum(fitnesses)
    for pop in range(population_size):
        parent = np.random.choice(population, p=prob)
        new_parameters = parent.params[pipeline_to_optimise].copy()
        for idx in range(new_parameters.shape[0]):
            new_parameters[idx] = np.clip(new_parameters[idx] + np.random.normal(loc=0, scale=sigmas[idx]), bounds[0][idx], bounds[1][idx])
        if N_models > 1:
            if np.random.rand() < change_model_p:
                model_idx = np.random.randint(0, N_models)
            else:
                model_idx = parent.model_idx
        else:
            model_idx = 0
        if resize_cfg["optimize"]:
            factor = np.clip(parent.factors[pipeline_to_optimise] + np.random.normal(0, resize_cfg["sigma"]), resize_cfg["min"], resize_cfg["max"])
        else:
            factor = parent.factors[pipeline_to_optimise]
        if optimize_order:
            if np.random.rand() < 0.5:
                order = rng.permuted(parent.order, axis=-1)
            else:
                order = parent.order
        else:
            order = parent.order

        all_new_params = parent.params.copy()
        all_new_params[pipeline_to_optimise] = new_parameters
        all_new_factors = parent.factors.copy()
        all_new_factors[pipeline_to_optimise] = factor
        new_population.append(Candidate(all_new_params, order, all_new_factors, pipeline_to_optimise))

    return new_population

# Samples random parameter values and orders
def initialize_population(population_size, mean, sigma, resize_cfg, param_shape, N_models, bounds, optimise_ensemble,
                          optimise_order, pipeline_to_optimise, params_sofar, factors_sofar):
    parameter_samples = np.random.normal(mean, sigma, size=(population_size, *param_shape))
    parameter_samples = np.clip(parameter_samples, bounds[0], bounds[1])
    rng = np.random.default_rng()
    order_samples = rng.permuted(np.tile(np.arange(param_shape[1]), (population_size, param_shape[0], 1)), axis=-1)
    if resize_cfg["optimize"]:
        factors = np.clip(np.random.normal(resize_cfg["init"], resize_cfg["sigma"], size=population_size), resize_cfg["min"], resize_cfg["max"])
    else:
        factors = np.ones(population_size)
    model_idx_samples = np.random.randint(0, N_models, size=population_size)
    new_population = []
    for pop in range(population_size):
        if optimise_order:
            order = order_samples[pop]
        else:
            order = np.tile(np.arange(param_shape[1]), (param_shape[0], 1))

        new_params = params_sofar.copy()
        new_factors = factors_sofar.copy()
        new_params[pipeline_to_optimise] = parameter_samples[pop]
        new_factors[pipeline_to_optimise] = factors[pop]
        new_population.append(Candidate(new_params, order, new_factors, pipeline_to_optimise)) # model_idx_samples[pop]))
    new_params = params_sofar.copy()
    new_factors = factors_sofar.copy()
    new_params[pipeline_to_optimise] = mean
    new_factors[pipeline_to_optimise] = 1.0
    new_population.append(Candidate(new_params, np.tile(np.arange(param_shape[1]), (param_shape[0], 1)), new_factors))
    return new_population

# Evaluates every candidate in the population on the problem
def evaluate_population(population, problem):
    for candidate in population:
        fitness = problem(*candidate())
        candidate.set_fitness(fitness)
    return population


# Selects the best candidates from the current population and offspring
def select(parents, population_size, offspring=None):
    if offspring == None:
        selection_pool = parents
    else:
        selection_pool = parents + offspring
    selection_pool.sort(key=lambda x: x.fitness)

    return selection_pool[0], selection_pool[:population_size]

class SimpleProblem:
    def __init__(
        self, model_lib, evaluator, pipeline_cfg, param_shape, augment_mask=True, use_both_lighting=False
    ):
        self.model_lib = model_lib
        self.N_models = len(model_lib)
        self.evaluator = evaluator
        self.pipeline_cfg = pipeline_cfg
        self.augment_mask = augment_mask
        self.use_both_lighting = use_both_lighting
        self.param_shape = param_shape

        n_var = len([arg for cfg in pipeline_cfg for fn in cfg for arg in fn["args"]])

    def __call__(self, parameters, order, resize_factor, model_idx):
        pipelines = []
        for i, param in enumerate(parameters):
            if param is not None:
                pipeline = ImagePipeline.build_pipeline(
                    self.pipeline_cfg[0],
                    param,
                    resize_factor[i],
                    self.augment_mask,
                    order,
                    use_both_lighting=self.use_both_lighting,
                )
                pipelines.append(pipeline)
            else:
                pipelines.append(None)

        score = self.evaluator.run(self.model_lib, pipelines)

        score = np.mean(score)

        return score


def optimize_Sequential_ES(
        pipeline_cfg,
        resize_cfg,
        model_lib,
        train_dataset,
        scoring_funcion,
        seed=0,
        augment_mask=True,
        use_pseudo_label=False,
        use_both_lighting=False,
        optimise_ensemble=False,
        optimise_order=False
):
#    pipeline_cfg = [cfg.augs] # if cfg.use_both_lighting else [cfg.augs]

    np.random.seed(seed=seed)
    torch.manual_seed(seed)

    evaluator = Evaluator(train_dataset, scoring_funcion, use_pseudo_label)

    init_vals = np.array([
        [arg["init"] for fn in cfg for arg in fn["args"]] for cfg in pipeline_cfg
    ])

    lbounds = np.array([
        [arg["min"] for fn in cfg for arg in fn["args"]] for cfg in pipeline_cfg
    ])
    ubounds = np.array([
        [arg["max"] for fn in cfg for arg in fn["args"]] for cfg in pipeline_cfg
    ])

    sigmas = np.array([
        [arg["sigma"] for fn in cfg for arg in fn["args"]] for cfg in pipeline_cfg
    ])

    resize_cfg = resize_cfg
    change_model_p = 0.5

    N_models = len(model_lib)
    param_shape = init_vals.shape

    bounds = (lbounds, ubounds)
    # Hyperparameters
    n_gen = N_models*10 #cfg.optimization["n_gen"]
    population_size = 15 #cfg.optimization["pop_size"]
    offspring_size = 15 #cfg.optimization["offspring_size"]

    problem = SimpleProblem(model_lib, evaluator, pipeline_cfg, param_shape, augment_mask, use_both_lighting)

    # Initialize and evaluate population
    population = initialize_population(population_size + offspring_size, init_vals, sigmas, resize_cfg,
                                       param_shape, N_models, bounds, optimise_ensemble, optimise_order, 0, [None]*N_models, [None]*N_models)

    # for pop in population:
    #     logging.info(f"individual with {pop.params} and {pop.factors}")

    population = evaluate_population(population, problem)
    best_candidate, population = select(population, population_size)

    logging.info(f"Generation: 0, best fitness: {best_candidate.fitness}, model: {best_candidate.model_idx}")

    pipeline_to_optimise = 0
    time_per_epoch = 0
    fitness_evolution = []
    for i in range(n_gen):
        start_epoch = time.time()
        offspring = sample_params(population, offspring_size, sigmas, resize_cfg, bounds, N_models, change_model_p, optimise_order, pipeline_to_optimise)
        offspring = evaluate_population(offspring, problem)
        best_candidate, population = select(population, population_size, offspring)

        for pop in population:
            logging.info(f"individual with {pop.params} and {pop.factors}")

        logging.info(
            f"Generation: {i+1}, best fitness: {best_candidate.fitness}, model: {best_candidate.model_idx}"
        )

        # Decrease step size
        sigmas *= 0.9
        change_model_p *= 0.9

        if ((i+1) % 10 == 0) and (i < n_gen-1):
            pipeline_to_optimise += 1
            population = initialize_population(population_size + offspring_size, init_vals, sigmas, resize_cfg,
                                       param_shape, N_models, bounds, optimise_ensemble, optimise_order, pipeline_to_optimise, best_candidate.params, best_candidate.factors)
            population = evaluate_population(population, problem)
            # for pop in population:
            #     logging.info(f"individual with {pop.params} and {pop.factors}")

        fitness_evolution.append(best_candidate.fitness)
        end_epoch = time.time()
        time_per_epoch += (end_epoch - start_epoch)

    print(f"Total time it took to do {n_gen} epochs: {time_per_epoch}")
    print(f"Time it took to do one epoch: {time_per_epoch/n_gen}")
    print(f"Fitness evolution: {fitness_evolution}")

    pipelines = []
    for i in range(N_models):
        pipeline = ImagePipeline.build_pipeline(
            pipeline_cfg[0], best_candidate.params[i], best_candidate.factors[i], augment_mask=augment_mask,
            order=best_candidate.order, use_both_lighting=use_both_lighting
        )
        pipelines.append(pipeline)
    return pipelines, best_candidate.model_idx
