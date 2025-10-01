"""
Utility functions for PUFBreaker.
"""
import numpy as np


def challenge_to_feature(challenge):
    """
    Transform binary challenge to parity feature vector.
    
    For Arbiter PUF: converts [c1, c2, ..., cn] to feature vector
    where each element represents cumulative parity.
    The feature vector includes a bias term (constant 1 at the end).
    
    Parameters
    ----------
    challenge : np.ndarray
        Binary challenge vector (0s and 1s) of shape (n_stages,)
        or (n_samples, n_stages)
    
    Returns
    -------
    np.ndarray
        Feature vector of shape (n_stages+1,) or (n_samples, n_stages+1)
    
    Examples
    --------
    >>> challenge = np.array([1, 0, 1, 1])
    >>> features = challenge_to_feature(challenge)
    >>> features.shape
    (5,)  # 4 stages + 1 bias
    """
    if challenge.ndim == 1:
        # Single challenge
        features = 1 - 2 * challenge  # Convert 0,1 to 1,-1
        # Add bias term
        features = np.append(features, 1.0)
        return features
    else:
        # Multiple challenges
        features = 1 - 2 * challenge
        # Add bias term to each challenge
        bias = np.ones((features.shape[0], 1))
        features = np.concatenate([features, bias], axis=1)
        return features


def compute_delay(features, delay_vector):
    """
    Compute delay difference for Arbiter PUF.
    
    Parameters
    ----------
    features : np.ndarray
        Feature vector(s) from challenge transformation
    delay_vector : np.ndarray
        PUF delay parameters
    
    Returns
    -------
    float or np.ndarray
        Delay difference(s)
    """
    if features.ndim == 1:
        return np.dot(features, delay_vector)
    else:
        return np.dot(features, delay_vector)


def add_noise(responses, noise_rate, rng=None):
    """
    Add noise to PUF responses.
    
    Parameters
    ----------
    responses : np.ndarray
        Binary responses (0s and 1s)
    noise_rate : float
        Probability of flipping each bit (0 to 1)
    rng : np.random.RandomState, optional
        Random number generator
    
    Returns
    -------
    np.ndarray
        Noisy responses
    """
    if rng is None:
        rng = np.random.RandomState()
    
    if noise_rate <= 0:
        return responses
    
    noise_mask = rng.rand(len(responses)) < noise_rate
    noisy_responses = responses.copy()
    noisy_responses[noise_mask] = 1 - noisy_responses[noise_mask]
    
    return noisy_responses


def generate_random_challenges(n_samples, n_stages, seed=None):
    """
    Generate random binary challenges.
    
    Parameters
    ----------
    n_samples : int
        Number of challenges to generate
    n_stages : int
        Number of stages (bits per challenge)
    seed : int, optional
        Random seed
    
    Returns
    -------
    np.ndarray
        Binary challenge matrix of shape (n_samples, n_stages)
    """
    rng = np.random.RandomState(seed)
    return rng.randint(0, 2, size=(n_samples, n_stages))


def accuracy_score(y_true, y_pred):
    """
    Compute classification accuracy.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    
    Returns
    -------
    float
        Accuracy between 0 and 1
    """
    return np.mean(y_true == y_pred)