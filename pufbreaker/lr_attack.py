"""
Logistic Regression attack on PUFs.

Works well on Arbiter PUFs but fails on XOR PUFs.
"""
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from .utils import challenge_to_feature, accuracy_score


class LRAttack:
    """
    Logistic Regression attack on PUFs.
    
    This attack works by learning the linear relationship between
    challenge features and responses. Very effective on Arbiter PUFs,
    but fails on XOR PUFs due to non-linearity.
    
    Parameters
    ----------
    C : float, optional
        Regularization parameter (default: 1.0)
    max_iter : int, optional
        Maximum iterations (default: 1000)
    
    Attributes
    ----------
    model : LogisticRegression
        The trained sklearn model
    train_accuracy : float
        Accuracy on training data
    training_time : float
        Time taken to train (seconds)
    
    Examples
    --------
    >>> from pufbreaker import ArbiterPUF, LRAttack
    >>> puf = ArbiterPUF(n_stages=64, seed=42)
    >>> X_train, y_train = puf.generate_dataset(1000)
    >>> X_test, y_test = puf.generate_dataset(500, seed=999)
    >>> 
    >>> attack = LRAttack()
    >>> attack.fit(X_train, y_train)
    >>> accuracy = attack.score(X_test, y_test)
    >>> print(f"Attack accuracy: {accuracy:.2%}")
    """
    
    def __init__(self, C=1.0, max_iter=1000):
        self.C = C
        self.max_iter = max_iter
        self.model = None
        self.train_accuracy = None
        self.training_time = None
        
    def fit(self, X_train, y_train):
        """
        Train the attack on challenge-response pairs.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training challenges (n_samples, n_stages)
        y_train : np.ndarray
            Training responses (n_samples,)
        
        Returns
        -------
        self
            Fitted attack model
        """
        start_time = time.time()
        
        # Transform challenges to features
        X_features = challenge_to_feature(X_train)
        
        # Train logistic regression
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver='lbfgs',
            random_state=42
        )
        self.model.fit(X_features, y_train)
        
        self.training_time = time.time() - start_time
        
        # Calculate training accuracy
        y_train_pred = self.predict(X_train)
        self.train_accuracy = accuracy_score(y_train, y_train_pred)
        
        return self
    
    def predict(self, X):
        """
        Predict responses for given challenges.
        
        Parameters
        ----------
        X : np.ndarray
            Challenges to predict
        
        Returns
        -------
        np.ndarray
            Predicted responses
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_features = challenge_to_feature(X)
        return self.model.predict(X_features)
    
    def predict_proba(self, X):
        """
        Predict response probabilities.
        
        Parameters
        ----------
        X : np.ndarray
            Challenges
        
        Returns
        -------
        np.ndarray
            Probability matrix (n_samples, 2)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_features = challenge_to_feature(X)
        return self.model.predict_proba(X_features)
    
    def score(self, X_test, y_test):
        """
        Calculate accuracy on test data.
        
        Parameters
        ----------
        X_test : np.ndarray
            Test challenges
        y_test : np.ndarray
            True test responses
        
        Returns
        -------
        float
            Test accuracy (0 to 1)
        """
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)
    
    def get_learned_weights(self):
        """
        Get the learned weight vector.
        
        Can be compared to true PUF delay vector for analysis.
        
        Returns
        -------
        np.ndarray
            Learned weights
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        return self.model.coef_[0]
    
    def __repr__(self):
        status = "trained" if self.model is not None else "untrained"
        return f"LRAttack(C={self.C}, {status})"