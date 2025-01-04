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
        ensemble, weights = [], []
        for i, n in enumerate(voting_weights):
            if n:
                ensemble.append(self.model_lib[i])
                weights.append(voting_weights[i])
        norm_weights = [float(w)/sum(weights) for w in weights]
        score = self.evaluator.run(ensemble, norm_weights, eval_type, sampler)
        return score
