import numpy as np
from numpy.linalg import inv

class RegularizedLinearRegressionEncoder:
    
    """A class for encoding categorical variables using a regularized linear regression model.
    
    Attributes:
        alpha (float, optional): The regularization strength; must be a positive float. The smaller the value, the stronger the regularization. Defaults to 0.1.
        l1_ratio (float, optional): The balance between L1 and L2 regularization. Must be between 0 and 1. The closer to 0, the more L2 regularization is applied. The closer to 1, the more L1 regularization is applied. Defaults to 0.5.
        categories_ (List[str]): A list of the categories in the data. Set after calling the `fit` method.
        encoded_columns_ (np.ndarray): A 2D numpy array with the encoded data. Set after calling the `transform` method.
        weights_ (np.ndarray): A 1D numpy array with the weights for each category. Set after calling the `fit` method.
        intercept_ (float): The intercept value for the linear regression model. Set after calling the `fit` method.
    """
    def __init__(self, alpha=0.1, l1_ratio=0.5):
        
        """Initialize the RegularizedLinearRegressionEncoder object.
        
        Args:
            alpha (float, optional): The regularization strength; must be a positive float. The smaller the value, the stronger the regularization. Defaults to 0.1.
            l1_ratio (float, optional): The balance between L1 and L2 regularization. Must be between 0 and 1. The closer to 0, the more L2 regularization is applied. The closer to 1, the more L1 regularization is applied. Defaults to 0.5.
        """
        
        
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.categories_ = None
        self.encoded_columns_ = None
        self.weights_ = None
        self.intercept_ = None
        
        
        
        
        
    def fit(self, X, y):
        """
        Fit the encoder to the input data.

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            The categorical data to be encoded.
        y : array-like, shape (n_samples,)
            The target values.

        Returns
        -------
        None
        """
            
        # Get the unique categories in the input data
        self.categories_ = list(set(X))
        
        # Encode the categories as one-hot vectors
        X_encoded = self.onehot_encode(X)
        
        # Add a column of ones to X_encoded to account for the intercept term
        X_encoded = np.hstack((np.ones((X_encoded.shape[0], 1)), X_encoded))
        
        # Calculate the regularization term
        reg_term = self.alpha * self.l1_ratio * np.ones(X_encoded.shape[1])
        
        # Use ridge regression to calculate the weights
        identity = np.identity(X_encoded.shape[1])
        identity[0, 0] = 0 # Do not regularize the intercept term
        self.weights_ = np.dot(inv(np.dot(X_encoded.T, X_encoded) + np.dot(reg_term, reg_term.T)), np.dot(X_encoded.T, y))
        
        # Save the encoded column names for later use
        self.encoded_columns_ = ["x_{}_{}".format(i, c) for i, c in enumerate(self.categories_)]
        
        
        
    def transform(self, X):
        """
        Encode the input categorical data.

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            The categorical data to be encoded.

        Returns
        -------
        encoded : array-like, shape (n_samples,)
            The encoded target values.
        """
            
        # Encode the categories as one-hot vectors
        X_encoded = self.onehot_encode(X)
        
        # Add a column of ones to X_encoded to account for the intercept term
        X_encoded = np.hstack((np.ones((X_encoded.shape[0], 1)), X_encoded))
        
        # Calculate the predicted target values
        return np.dot(X_encoded, self.weights_)
    
    def onehot_encode(self, X):
        """
        Encode categorical data as one-hot vectors.
        
        Parameters
        ----------
        X : array-like, shape (n_samples,)
            Input data, where n_samples is the number of samples.
        
        Returns
        -------
        X_encoded : array-like, shape (n_samples, n_features)
            Encoded input data, where n_features is the number of features.
        """
        X_encoded = np.zeros((len(X), len(self.categories_)))
        for i, x in enumerate(X):
            X_encoded[i, self.categories_.index(x)] = 1
        return X_encoded
    
    def fit_transform(self, X, y):
        """
        Fit the encoding model to the input data and return the transformed data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples,)
            Input data, where n_samples is the number of samples.
        y : array-like, shape (n_samples,)
            Target values.
        
        Returns
        -------
        X_encoded : array-like, shape (n_samples, n_features)
            Encoded input data, where n_features is the number of features.
        """
        self.fit(X, y)
        return self.transform(X)
