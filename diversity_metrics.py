import numpy as np
from scipy import stats
from conf_mat import ConfusionMatrix


def pierson_correlation(probabilities):
    coeff_dict = {}
    for i, _ in enumerate(probabilities):
        for j in range(i+1, len(probabilities)):
#            print(probabilities[i], probabilities[j])
#            print("-"*20)
            res = stats.pearsonr(probabilities[i], probabilities[j])
            coeff_dict[(i,j)] = res
    return coeff_dict


def confmat_miou(out1, out2):
    # TODO i have already flattened the probabilities saved in the npy file
    # so we cant just do this :D either have to unflatten or to store them anew
    confmat = ConfusionMatrix(21)
    for i, probs in enumerate(out1):
        print(probs)
        print(out2[i])
        exit(0)
        confmat.update(probs.flatten(), out2[i].flatten())
    return confmat


def diversity_segmentors(model1, model2, name):
    probs1 = np.load(f'/dataB3/nadia_dobreva/model{model1}_preds.npy')[:1000]
    probs2 = np.load(f'/dataB3/nadia_dobreva/model{model2}_preds.npy')[:1000]

    if name == "pierson":
        pierson_dict = pierson_correlation([probs1, probs2])
        print(pierson_dict)
    if name == "miou":
        confmat = confmat_miou(probs1, probs2)
        print(confmat)


if __name__ == "__main__":
    diversity_segmentors(4, 5, "pierson")
