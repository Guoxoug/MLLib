import numpy as np
import torch
from torch.nn import functional as F
from scipy.special import softmax

""" Numpy Implementation of Uncertainty Measures """

def generate_prob_np(d1, d2, d3):
    return softmax(np.random.rand(d1, d2, d3), axis=2)

def kl_divergence_np(probs1, probs2, epsilon=1e-10):
    return np.sum(probs1 * (np.log(probs1 + epsilon) - np.log(probs2 + epsilon)), axis=1)

def kl_divergence_torch(probs1, probs2, epsilon=1e-10):
    return torch.sum(probs1 * (torch.log(probs1 + epsilon) - torch.log(probs2 + epsilon)), dim=1)

def expected_pairwise_kl_divergence_np(probs, epsilon=1e-10):
    kl = 0.0
    for i in range(probs.shape[1]):
        for j in range(probs.shape[1]):
            kl += kl_divergence_np(probs[:, i, :], probs[:, j, :], epsilon)
    return kl

def expected_pairwise_kl_divergence_torch(probs, epsilon=1e-10):
    kl = 0.0
    for i in range(probs.shape[1]):
        for j in range(probs.shape[1]):
            kl += kl_divergence_torch(probs[:, i, :], probs[:, j, :], epsilon)
    return kl


def entropy_of_expected_np(probs, epsilon=1e-10):
    mean_probs = np.mean(probs, axis=1)
    log_probs = -np.log(mean_probs + epsilon)
    return np.sum(mean_probs * log_probs, axis=1)

def entropy_of_expected_torch(probs, epsilon=1e-10):
    mean_probs = torch.mean(probs, dim=1)
    log_probs = -torch.log(mean_probs + epsilon)
    return torch.sum(mean_probs * log_probs, dim=1)

def entropy_np(probs, epsilon=1e-10):
    log_probs = -np.log(probs + epsilon)
    return np.sum(probs * log_probs, axis=1)

def entropy_torch(probs, epsilon=1e-10):
    log_probs = -torch.log(probs + epsilon)
    return torch.sum(probs * log_probs, dim=1) 

def expected_entropy_np(probs, epsilon=1e-10):
    log_probs = -np.log(probs + epsilon)
    return np.mean(np.sum(probs * log_probs, axis=2), axis=1)

def expected_entropy_torch(probs, epsilon=1e-10):
    log_probs = -torch.log(probs + epsilon)
    return torch.mean(torch.sum(probs * log_probs, dim=2), dim=1)

def mutual_information_np(probs, epsilon=1e-10):
    eoe = entropy_of_expected_np(probs, epsilon)
    exe = expected_entropy_np(probs, epsilon)
    return eoe - exe

def mutual_information_torch(probs, epsilon=1e-10):
    eoe = entropy_of_expected_torch(probs, epsilon)
    exe = expected_entropy_torch(probs, epsilon)
    return eoe - exe

def ensemble_uncertainties_np(probs, epsilon=1e-10):
    mean_probs = np.mean(probs, axis=1)
    conf = np.max(mean_probs, axis=1)

    eoe = entropy_of_expected_np(probs, epsilon)
    exe = expected_entropy_np(probs, epsilon)
    mutual_info = eoe - exe

    epkl = expected_pairwise_kl_divergence_np(probs, epsilon)

    uncertainty = {'confidence': conf,
                   'entropy_of_expected': eoe,
                   'expected_entropy': exe,
                   'mutual_information': mutual_info,
                   'EPKL': epkl}

    return uncertainty

def ensemble_uncertainties_torch(probs, epsilon=1e-10):
    mean_probs = torch.mean(probs, dim=1)
    conf, _ = torch.max(mean_probs, dim=1)

    eoe = entropy_of_expected_torch(probs, epsilon)
    exe = expected_entropy_torch(probs, epsilon)
    mutual_info = eoe - exe

    epkl = expected_pairwise_kl_divergence_torch(probs, epsilon)

    uncertainty = {'confidence': conf,
                   'entropy_of_expected': eoe,
                   'expected_entropy': exe,
                   'mutual_information': mutual_info,
                   'EPKL': epkl}

    return uncertainty
