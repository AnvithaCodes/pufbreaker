"""
Multi-Layer Perceptron (Neural Network) attack on PUFs.

Can attack both Arbiter and XOR PUFs due to non-linear capability.
"""
import numpy as np
import time
from sklearn.neural_network import MLPClassifier
from .utils import challenge_to_feature, accuracy_score


class MLPAttack:
    """
    Neural Network attack on PUFs.
    
    Uses multi-layer perceptron to learn non-linear relationships.
    Essential for attacking XOR PUFs where linear models fail.
    
    Parameters
    ----------
    hidden_layers : tuple, optional
        Hidden layer sizes (default: (128, 64))
    max_iter : int, optional
        Maximum training epochs (default: 500)
    learning_rate_init : float, optional
        Initial learning rate (default: 0.001)
    
    Attributes
    ----------
    model : MLPClassifier
        The trained neural network
    train_accuracy : float
        Training accuracy
    training_time : float
        Training time in seconds
    
    Examples
    --------
    >>> from pufbreaker import XORPUF, MLPAttack
    >>> xor_puf = XORPUF(n_stages=64, k=2, seed=42)
    >>> X_train, y_train = xor_puf.generate_dataset(5000)
    >>> X_test, y_test = xor_puf.generate_dataset(1000, seed=999)
    >>> 
    >>> attack = MLPAttack(hidden_layers=(128, 64))
    >>> attack.fit(X_train, y_train)
    >>> accuracy = attack.score(X_test, y_test)
    >>> print(f"Neural network accuracy: {accuracy:.2%}")
    """
    
    def __init__(self, hidden_layers=(128, 64), max_iter=500, 
                 learning_rate_init=0.001):
        self.hidden_layers = hidden_layers
        self.max_iter = max_iter
        self.learning_rate_init = learning_rate_init
        self.model = None
        self.train_accuracy = None
        self.training_time = None
        
    def fit(self, X_train, y_train, verbose=False):
        """
        Train the neural network attack.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training challenges (n_samples, n_stages)
        y_train : np.ndarray
            Training responses (n_samples,)
        verbose : bool, optional
            Print training progress
        
        Returns
        -------
        self
            Fitted attack model
        """
        start_time = time.time()
        
        # Transform challenges to features
        X_features = challenge_to_feature(X_train)
        
        # Create and train neural network
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layers,
            max_iter=self.max_iter,
            learning_rate_init=self.learning_rate_init,
            solver='adam',
            activation='relu',
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=verbose
        )
        
        self.model.fit(X_features, y_train)
        
        self.training_time = time.time() - start_time
        
        # Calculate training accuracy
        y_train_pred = self.predict(X_train)
        self.train_accuracy = accuracy_score(y_train, y_train_pred)
        
        return self
    
    def predict(self, X):
        """
        Predict responses for challenges.
        
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
        Calculate test accuracy.
        
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
    
    def get_training_history(self):
        """
        Get loss curve from training.
        
        Returns
        -------
        dict
            Training history with loss values
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        return {
            'loss_curve': self.model.loss_curve_,
            'n_iter': self.model.n_iter_,
            'best_loss': self.model.best_loss_
        }
    
    def __repr__(self):
        status = "trained" if self.model is not None else "untrained"
        return f"MLPAttack(hidden_layers={self.hidden_layers}, {status})"