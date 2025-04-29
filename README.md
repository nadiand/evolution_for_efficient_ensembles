# Efficient Test-time Data Augmentation Pipelines and Ensemble Selection Using Evolution

This repository contains the implementations of a number of evolution-powered algorithms for tackling data shifts at test-time. Although the majority of these approaches do not have a significant difference in performance, they outperform two baselines (a single model, and the ensemble consisting all models in the library) on two benchmark datasets (CIFAR-100, PascalVOC) and are relatively time-efficient. Furthermore, they remove the need for training or accessing the train data.

The algorithms are as follows: 
- ENSEC: evolutionary algorithm for ensemble selection, given a library of pre-trained models and a target task. Implemented in `ensemble_selection_weights.py`.
- DAP: evolutionary algorithm for test-time data augmentation pipeline optimization given a target task. This is a novel approach resulting in a pipeline of image transformations aimed to reverse the present data shift. Implemented in `pipeline_evolution.py`.
- ENSDAP(gen), ENSDAP(ind), DAPENS: three algorithms involving the previously described two, that both optimize ensembles and shift-reversing pipelines, given a library of pre-trained models and a target task. ENSDAP(gen) and DAPENS result in an ensemble and a single general data augmentation pipeline to be applied to all models in it (with the only difference being in the order of the optimization of the two), while ENSDAP(ind) results in an ensemble and an individual pipeline for every model in it. 

For more details on the algorithms, the conducted experiments and achieved results, please refer to the paper also included in the repository.
