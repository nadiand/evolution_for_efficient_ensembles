import numpy as np
import logging
from pipeline_evaluator import Evaluator
from image_pipeline import ImagePipeline

#Candidate class that contains the parameters and order
class Candidate:
    def __init__(self, params, order, factor = 1, model_idx = 0):
        self.params = params
        self.order = order
        self.factor = factor
        self.model_idx = model_idx
        self.fitness = None

    #Evaluates the candidates parameters in the given order
    def __call__(self):
        return self.params, self.order, self.factor, self.model_idx

    def set_fitness(self, fitness):
        self.fitness = fitness

#Samples a population that optimizes only the parameter of one function
def sample_param(population, index, sigmas, population_size, bounds):
    new_population = []
    fitnesses = np.array([-p.fitness for p in population])
    prob = fitnesses/np.sum(fitnesses)
    for pop in range(population_size):
        parent = np.random.choice(population, p=prob)
        new_parameters = parent.params.copy()
        for idx in range(new_parameters.shape[0]):
            new_parameters[idx, index] = np.clip(new_parameters[idx, index] + np.random.normal(loc=0, scale=sigmas[idx, index]), bounds[0][idx, index], bounds[1][idx, index])
        new_population.append(Candidate(new_parameters, parent.order, parent.factor, parent.model_idx))

    return new_population

def sample_params(population, population_size, sigmas, bounds):
    new_population = []
    fitnesses = np.array([-p.fitness for p in population])
    prob = fitnesses/np.sum(fitnesses)
    for pop in range(population_size):
        parent = np.random.choice(population, p=prob)
        new_parameters = parent.params.copy()
        for idx in range(new_parameters.shape[0]):
            new_parameters[idx] = np.clip(new_parameters[idx] + np.random.normal(loc=0, scale=sigmas[idx]), bounds[0][idx], bounds[1][idx])
        new_population.append(Candidate(new_parameters, parent.order, parent.factor, parent.model_idx))

    return new_population

#Samples a small population that optimizes the order of the functions
def change_order(population, population_size):
    new_population = []
    fitnesses = np.array([-p.fitness for p in population])
    prob = fitnesses/np.sum(fitnesses)
    rng = np.random.default_rng()
    for pop in range(population_size):
        candidate = np.random.choice(population, p=prob)
        new_population.append(Candidate(candidate.params, rng.permuted(candidate.order, axis=-1), candidate.factor, candidate.model_idx))

    return new_population

def change_model(population, population_size, N_models):
    new_population = []
    fitnesses = np.array([-p.fitness for p in population])
    prob = fitnesses/np.sum(fitnesses)
    for pop in range(population_size):
        candidate = np.random.choice(population, p=prob)
        new_population.append(Candidate(candidate.params, candidate.order, candidate.factor, np.random.randint(0, N_models)))

    return new_population

def change_factor(population, population_size, sigma):
    new_population = []
    fitnesses = np.array([-p.fitness for p in population])
    prob = fitnesses/np.sum(fitnesses)
    for pop in range(population_size):
        candidate = np.random.choice(population, p=prob)
        new_factor = np.clip(candidate.factor + np.random.normal(0, sigma), 0.3, 3.0)
        new_population.append(Candidate(candidate.params, candidate.order, new_factor, candidate.model_idx))

    return new_population

#Samples random parameter values and orders
def initialize_population(population_size, mean, sigma, resize_sigma, param_shape, N_models, bounds, optimise_ensemble, optimise_order):
    parameter_samples = np.random.normal(mean, sigma, size=(population_size,*param_shape))
    parameter_samples = np.clip(parameter_samples, bounds[0], bounds[1])
    rng = np.random.default_rng()
    order_samples = rng.permuted(np.tile(np.arange(param_shape[1]),(population_size, param_shape[0], 1)), axis=-1)
    factors = np.clip(np.random.normal(1, resize_sigma, size=population_size), 0.3, 3)
    model_idx_samples = np.random.randint(0, N_models, size=population_size)
    new_population = []
    for pop in range(population_size):
        if optimise_order:
            order = order_samples[pop]
        else:
            order = np.tile(np.arange(param_shape[1]),(param_shape[0], 1))
        if N_models==1:
            new_population.append(Candidate(parameter_samples[pop], order, factors[pop]))
        elif optimise_ensemble:
            new_population.append(Candidate(parameter_samples[pop], order, factors[pop], -1))
        else:
            new_population.append(Candidate(parameter_samples[pop], order, factors[pop], model_idx_samples[pop]))
    return new_population

#Evaluates every candidate in the population on the problem
def evaluate_population(population, problem):
    for candidate in population:
        fitness = problem(*candidate())
        candidate.set_fitness(fitness)
    return population

#Selects the best candidates from the current population and offspring
def select(parents, population_size, offspring=None):
    if offspring==None:
        selection_pool = parents
    else:
        selection_pool = parents + offspring
    selection_pool.sort(key=lambda x: x.fitness)

    return selection_pool[0], selection_pool[:population_size]

class SimpleProblem:
    def __init__(
        self, model_lib, evaluator, pipeline_cfg, augment_mask=True, use_both_lighting=False
    ):
        self.model_lib = model_lib
        self.evaluator = evaluator
        self.pipeline_cfg = pipeline_cfg
        self.augment_mask = augment_mask
        self.use_both_lighting = use_both_lighting

        n_var = len([arg for cfg in pipeline_cfg for fn in cfg for arg in fn["args"]])

    def __call__(self, parameters, order, factor, model_idx):
        pipeline = ImagePipeline.build_pipeline(
            self.pipeline_cfg[0],
            parameters,
            factor,
            self.augment_mask,
            order,
#            select_model=False,
            use_both_lighting=self.use_both_lighting,
        )

        if model_idx==-1:
            model_lib = self.model_lib
        else:
            model_lib = [self.model_lib[model_idx]]

        score = self.evaluator.run(model_lib, pipeline)
        score = np.mean(score)

        return score


def optimize_Sequential_ES(
    pipeline_cfg,
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
    init_sigmas = np.array([
        [arg["init_sigma"] for fn in cfg for arg in fn["args"]] for cfg in pipeline_cfg
    ])
    sigmas = np.array([
        [arg["sigma"] for fn in cfg for arg in fn["args"]] for cfg in pipeline_cfg
    ])

    resize_sigma = 0.2

    N_models = len(model_lib)
    param_shape = init_vals.shape

    bounds = (lbounds, ubounds)
    #Hyperparameters
    n_gen = 2 #20 #optimization_cfg["n_gen"]
    population_size = 6 #15 #optimization_cfg["pop_size"]
    offspring_size = 6 #15 #optimization_cfg["offspring_size"]

    problem = SimpleProblem(model_lib, evaluator, pipeline_cfg, augment_mask, use_both_lighting)

    #Initialize and evaluate population
    population = initialize_population(population_size + offspring_size, init_vals, init_sigmas, resize_sigma, param_shape, N_models, bounds, optimise_ensemble, optimise_order)
    population = evaluate_population(population, problem)
    best_candidate, population = select(population, population_size)

    for i in range(n_gen):
        if N_models>1 and not optimise_ensemble:
            offspring = change_model(population, offspring_size, N_models)
            offspring = evaluate_population(offspring, problem)
            best_candidate, population = select(population, population_size, offspring)

        # Optimize the order for one generation
        if param_shape[1]>1 and optimise_order:
            offspring = change_order(population, offspring_size)
            offspring = evaluate_population(offspring, problem)
            best_candidate, population = select(population, population_size, offspring)

        offspring = change_factor(population, offspring_size, resize_sigma)
        offspring = evaluate_population(offspring, problem)
        best_candidate, population = select(population, population_size, offspring)

        offspring = sample_params(population, offspring_size, sigmas, bounds)
        offspring = evaluate_population(offspring, problem)
        best_candidate, population = select(population, population_size, offspring)

        logging.info(f"Generation: {i+1}, best fitness: {best_candidate.fitness}, order: {best_candidate.order}, factor: {best_candidate.factor}, model: {best_candidate.model_idx}")

    pipeline = ImagePipeline.build_pipeline(
        pipeline_cfg[0], best_candidate.params, best_candidate.factor, augment_mask=augment_mask, order = best_candidate.order, use_both_lighting=use_both_lighting
    )
    logging.info(f"Best candidate's statistics:\nParams: {best_candidate.params}\nFactor: {best_candidate.factor}\nOrder: {best_candidate.order}")
    return pipeline, best_candidate.model_idx
