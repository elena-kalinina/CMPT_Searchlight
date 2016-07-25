# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 16:04:33 2016

@author: Fabian Pedregosa
"""

import numpy as np
import itertools
from joblib import Parallel, delayed, cpu_count




def test_permutation(activation_img, condition, modality, subj,
                     verbose=True, n_perm=10000, n_jobs=1, random_seed=0):
    """
    Parameters
    ==========
    activation_img: array, shape (n_samples, n_features)
        The activation image (aka beta-map)

    condition: integer array, shape (n_samples,)
        The condition for every sample.

    modality: integer array, shape (n_samples,)
        Modality for each sample. For now this is limited to two
        modalities.

    subj: integer array, shape (n_samples,)
        Array indicating to which subject do the activation_img
        correspond. Correlations will only be computed within
        the same subject. In case all images correspond to a single
        subject set this array to a constant value.

    n_perm : integer
        number of permutations.

    n_jobs: integer
        number of processors to use (-1 for all).

    Returns
    =======
    pval: float
    T0 : test statistic
    T_perm : all permuted statistics
    """
    np.random.seed(random_seed)
    condition = np.array(condition)
    modality = np.array(modality)
    subj = np.array(subj)

    # check the data
    n_samples, n_features = activation_img.shape
    assert len(condition) == n_samples
    assert len(modality) == n_samples

    unique_modalities = np.unique(modality)
    unique_conditions = np.unique(condition)
    assert len(unique_modalities) == 2

    # number of subjects
    unique_sub = np.unique(subj)

    if verbose:
        print('%s subjects were given' % unique_sub.size)

    # compute test statistic
    T0 = 0.0
    for s in unique_sub:
        idx_sub = (subj == s)
        img_cond_modality_1 = []
        img_cond_modality_2 = []
        idx_mod_1 = (modality[idx_sub] == unique_modalities[0])
        idx_mod_2 = (modality[idx_sub] == unique_modalities[1])
        for cond in unique_conditions:
            idx_cond_m1 = (condition == cond) & idx_mod_1
            idx_cond_m2 = (condition == cond) & idx_mod_2
            img_cond_modality_1.append(activation_img[idx_cond_m1].mean(0))
            img_cond_modality_2.append(activation_img[idx_cond_m2].mean(0))
        img_cond_modality_1 = np.array(img_cond_modality_1)
        img_cond_modality_2 = np.array(img_cond_modality_2)

        T0 += test_stat(img_cond_modality_1, img_cond_modality_2)

    # compute the permuted statistic
    # all_permutations =
    idx_2 = np.arange(n_samples)[modality == unique_modalities[0]]
    all_permutations = [np.random.permutation(idx_2.size) for _ in range(n_perm)]

    n_splits = n_jobs
    if n_splits < 0:
        n_splits = cpu_count()
    all_permutations = np.array_split(all_permutations, n_splits)

    all_T_perm = Parallel(n_jobs=n_jobs)(
        delayed(_compute_perms)(
            perm, n_samples, modality, unique_modalities, condition,
            unique_sub, subj, unique_conditions, activation_img)
        for perm in all_permutations)

    T_perm = np.array(all_T_perm).ravel()
    pval = 1 - (T0 >= T_perm).mean()
    return pval, T0, T_perm


# helper routine for parallelization
def _compute_perms(perm, n_samples, modality, unique_modalities, condition,
                   unique_sub, subj, unique_conditions, activation_img):
    T_perm = []
    for p in perm:
        T_subj = 0.0
        idx_2 = np.arange(n_samples)[modality == unique_modalities[0]]
        condition_perm = condition.copy()
        condition_perm[idx_2] = condition_perm[idx_2][p]
        for s in unique_sub:
            idx_sub = (subj == s)
            img_cond_modality_1 = []
            img_cond_modality_2 = []
            idx_mod_1 = (modality[idx_sub] == unique_modalities[0])
            idx_mod_2 = (modality[idx_sub] == unique_modalities[1])
            for cond in unique_conditions:
                idx_1 = (condition_perm == cond) & idx_mod_1
                idx_2 = (condition_perm == cond) & idx_mod_2
                img_cond_modality_1.append(activation_img[idx_1].mean(0))
                img_cond_modality_2.append(activation_img[idx_2].mean(0))
            img_cond_modality_1 = np.array(img_cond_modality_1)
            img_cond_modality_2 = np.array(img_cond_modality_2)
            T_subj += test_stat(img_cond_modality_1, img_cond_modality_2)

        T_perm.append(T_subj)

    return T_perm

# test statistic
def test_stat(img_cond_modality_1, img_cond_modality_2):
    """
    This is the test statistic used by the permutation test

    Parameters
    ==========
    img_cond_modality_1 : array, shape (n_conditions, n_features)

    img_cond_modality_2 : array, shape (n_conditions, n_features)
    """
    n_conditions = img_cond_modality_1.shape[0]
    conditions = np.arange(n_conditions)

    # initialize
    within_condition_t = 0
    within_condition_counter = 0
    cross_condition_t = 0
    cross_condition_counter = 0

    # generate all pairwise comparisons
    for (a, b) in itertools.product(conditions, conditions):
        A = img_cond_modality_1[a]
        B = img_cond_modality_2[b]
        if a == b:
            within_condition_counter += 1
            within_condition_t += np.corrcoef(A, B)[0, 1]
        else:
            cross_condition_counter += 1
            cross_condition_t += np.corrcoef(A, B)[0, 1]
    assert within_condition_counter > 0
    assert cross_condition_counter > 0
    ret = within_condition_t / float(within_condition_counter) - \
            cross_condition_t / float(cross_condition_counter)
    assert np.isfinite(ret)
    return ret


def generate_synthetic_data(n_samples, width, noise_amplitude=0.5, only_noise=False):

    if only_noise:
        w_A1 = np.zeros((width, width))
        w_A2 = np.zeros((width, width))
        w_B1 = np.zeros((width, width))
        w_B2 = np.zeros((width, width))
    else:
        w_A1 = np.zeros((width, width))
        w_A1[20:30, 20:30] = 1.
        w_A1[160:170, 160:170] = 1.

        w_A2 = np.zeros((width, width))
        w_A2[20:30, 20:30] = 1.
        #     w_A2[50:60, 20:30] = 1.
        w_A2[20:30, 160:170] = 1.
        #     w_A2[160:170, 20:30] = 1.

        w_B1 = np.zeros((width, width))
        w_B1[160:170, 20:30] = 1.
        #     w_B1[50:60, 20:30] = 1.
        w_B1[160:170, 160:170] = 1.
        w_B1[20:30, 160:170] = 1.

        w_B2 = np.zeros((width, width))
        w_B2[160:170, 20:30] = 1.
        w_B2[20:30, 160:170] = 1.

    ground_truth = (w_A1, w_A2, w_B1, w_B2)

    samples_A1 = []
    for i in range(n_samples):
        tmp = w_A1 + noise_amplitude * np.random.randn(*w_A1.shape)
        tmp = tmp.ravel()
        # tmp -= np.mean(tmp)
        # tmp /= np.std(tmp)
        samples_A1.append(tmp.ravel())

    samples_A2 = []
    for i in range(n_samples):
        tmp = w_A2 + noise_amplitude * np.random.randn(*w_A1.shape)
        tmp = tmp.ravel()
        # tmp -= np.mean(tmp)
        # tmp /= np.std(tmp)
        samples_A2.append(tmp.ravel())

    samples_B1 = []
    for i in range(n_samples):
        tmp = w_B1 + noise_amplitude * np.random.randn(*w_A1.shape)
        tmp = tmp.ravel()
        # tmp -= np.mean(tmp)
        # tmp /= np.std(tmp)
        samples_B1.append(tmp.ravel())

    samples_B2 = []
    for i in range(n_samples):
        tmp = w_B2 + noise_amplitude * np.random.randn(*w_A1.shape)
        tmp = tmp.ravel()
        # tmp -= np.mean(tmp)
        # tmp /= np.std(tmp)
        samples_B2.append(tmp.ravel())


    samples = np.concatenate((samples_A1, samples_A2, samples_B1, samples_B2), axis=0)
    assert np.isnan(samples).sum() == 0

    condition = [0] * n_samples + [1] * n_samples + [0] * n_samples + [1] * n_samples
    modality = [0] * (2 * n_samples) + [1] * (2 * n_samples)
    return ground_truth, samples, condition, modality
