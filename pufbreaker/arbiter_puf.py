"""
Arbiter PUF implementation.

An Arbiter PUF consists of two parallel delay chains. At each stage,
the challenge bit determines whether signals cross or go straight.
The response depends on which signal reaches the arbiter first.
"""
import numpy as np
from .utils import challenge_to_feature, compute_delay, add_noise


class ArbiterPUF:
    """
    Arbiter PUF simulator.
    
    The Arbiter PUF is the simplest delay-based PUF. It can be attacked
    efficiently with linear machine learning models.
    
    Parameters
    ----------
    n_stages : int
        Number of stages in the delay chain (typically 64 or 128)
    noise : float, optional
        Noise probability (0 to 1). Default is 0.01 (1% noise)
    seed : int, optional
        Random seed for reproducibility
    
    Attributes
    ----------
    delay_vector : np.ndarray
        The secret delay parameters that define this PUF instance
    
    Examples
    --------
    >>> puf = ArbiterPUF(n_stages=64, noise=0.01, seed=42)
    >>> challenges, responses = puf.generate_dataset(n_samples=1000)
    >>> print(f"Generated {len(responses)} CRPs")
    """
    
    def __init__(self, n_stages=64, noise=0.01, seed=None):
        self.n_stages = n_stages
        self.noise = noise
        self.rng = np.random.RandomState(seed)
        
        # Generate random delay vector (this is the PUF's "secret")
        # In real hardware, this comes from manufacturing variation
        self.delay_vector = self.rng.randn(n_stages + 1)
        
    def evaluate(self, challenge):
        """
        Evaluate PUF on a single challenge.
        
        Parameters
        ----------
        challenge : np.ndarray
            Binary challenge vector of length n_stages
        
        Returns
        -------
        int
            Response bit (0 or 1)
        """
        features = challenge_to_feature(challenge)
        delay = compute_delay(features, self.delay_vector)
        
        # Response is based on sign of delay difference
        response = 1 if delay >= 0 else 0
        
        # Add noise
        if self.noise > 0 and self.rng.rand() < self.noise:
            response = 1 - response
        
        return response
    
    def evaluate_batch(self, challenges):
        """
        Evaluate PUF on multiple challenges (faster).
        
        Parameters
        ----------
        challenges : np.ndarray
            Binary challenge matrix of shape (n_samples, n_stages)
        
        Returns
        -------
        np.ndarray
            Response vector of length n_samples
        """
        features = challenge_to_feature(challenges)
        delays = compute_delay(features, self.delay_vector)
        
        # Convert delays to binary responses
        responses = (delays >= 0).astype(int)
        
        # Add noise
        responses = add_noise(responses, self.noise, self.rng)
        
        return responses
    
    def generate_dataset(self, n_samples=1000, seed=None):
        """
        Generate a dataset of challenge-response pairs.
        
        Parameters
        ----------
        n_samples : int
            Number of CRPs to generate
        seed : int, optional
            Random seed for challenge generation
        
        Returns
        -------
        challenges : np.ndarray
            Challenge matrix of shape (n_samples, n_stages)
        responses : np.ndarray
            Response vector of length n_samples
        """
        rng = np.random.RandomState(seed)
        challenges = rng.randint(0, 2, size=(n_samples, self.n_stages))
        responses = self.evaluate_batch(challenges)
        
        return challenges, responses
    
    def get_delay_vector(self):
        """
        Get the delay vector (for analysis/testing only).
        
        In real hardware, this would be unknown to attackers.
        
        Returns
        -------
        np.ndarray
            The delay vector
        """
        return self.delay_vector.copy()
    
    def __repr__(self):
        return f"ArbiterPUF(n_stages={self.n_stages}, noise={self.noise})"