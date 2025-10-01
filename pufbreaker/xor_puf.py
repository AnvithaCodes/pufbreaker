"""
XOR PUF implementation.

An XOR PUF combines k parallel Arbiter PUFs using XOR operation.
This creates non-linearity that resists linear ML attacks.
"""
import numpy as np
from .arbiter_puf import ArbiterPUF


class XORPUF:
    """
    XOR PUF simulator.
    
    Combines k Arbiter PUFs with XOR to increase security.
    Linear attacks fail on XOR PUFs, requiring neural networks.
    
    Parameters
    ----------
    n_stages : int
        Number of stages per Arbiter PUF
    k : int
        Number of parallel Arbiter PUFs to XOR (typically 2-5)
    noise : float, optional
        Noise probability per Arbiter PUF
    seed : int, optional
        Random seed
    
    Examples
    --------
    >>> xor_puf = XORPUF(n_stages=64, k=2, noise=0.01, seed=42)
    >>> challenges, responses = xor_puf.generate_dataset(5000)
    >>> print(f"XOR-{xor_puf.k} PUF with {xor_puf.n_stages} stages")
    """
    
    def __init__(self, n_stages=64, k=2, noise=0.01, seed=None):
        self.n_stages = n_stages
        self.k = k
        self.noise = noise
        self.rng = np.random.RandomState(seed)
        
        # Create k independent Arbiter PUFs
        self.arbiters = []
        for i in range(k):
            # Each arbiter gets a different seed
            sub_seed = None if seed is None else seed + i
            arbiter = ArbiterPUF(n_stages=n_stages, noise=noise, seed=sub_seed)
            self.arbiters.append(arbiter)
    
    def evaluate(self, challenge):
        """
        Evaluate XOR PUF on a single challenge.
        
        Parameters
        ----------
        challenge : np.ndarray
            Binary challenge vector
        
        Returns
        -------
        int
            XOR of all arbiter responses (0 or 1)
        """
        # Get response from each arbiter and XOR them
        responses = [arbiter.evaluate(challenge) for arbiter in self.arbiters]
        xor_response = np.bitwise_xor.reduce(responses)
        
        return int(xor_response)
    
    def evaluate_batch(self, challenges):
        """
        Evaluate XOR PUF on multiple challenges.
        
        Parameters
        ----------
        challenges : np.ndarray
            Challenge matrix of shape (n_samples, n_stages)
        
        Returns
        -------
        np.ndarray
            Response vector
        """
        # Get responses from all arbiters
        all_responses = np.array([
            arbiter.evaluate_batch(challenges) 
            for arbiter in self.arbiters
        ])
        
        # XOR them together
        xor_responses = np.bitwise_xor.reduce(all_responses, axis=0)
        
        return xor_responses
    
    def generate_dataset(self, n_samples=1000, seed=None):
        """
        Generate challenge-response pairs.
        
        Parameters
        ----------
        n_samples : int
            Number of CRPs
        seed : int, optional
            Random seed
        
        Returns
        -------
        challenges : np.ndarray
            Challenge matrix
        responses : np.ndarray
            Response vector
        """
        rng = np.random.RandomState(seed)
        challenges = rng.randint(0, 2, size=(n_samples, self.n_stages))
        responses = self.evaluate_batch(challenges)
        
        return challenges, responses
    
    def get_delay_vectors(self):
        """
        Get delay vectors from all Arbiter PUFs.
        
        Returns
        -------
        list of np.ndarray
            List of delay vectors
        """
        return [arbiter.get_delay_vector() for arbiter in self.arbiters]
    
    def __repr__(self):
        return f"XORPUF(n_stages={self.n_stages}, k={self.k}, noise={self.noise})"