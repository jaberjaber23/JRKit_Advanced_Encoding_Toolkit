import numpy as np

    
class RBFEncoder:
    
    """Radial Basis Function (RBF) Encoder

    This class implements a radial basis function (RBF) encoding for transforming
    a multi-dimensional input array into a new representation in a higher-dimensional
    space. The RBF encoding is computed as the weighted sum of Gaussian functions
    centered at a specified set of basis functions.

    Parameters
    ----------
    centers : array-like, shape (n_centers, n_features)
        The centers of the basis functions.
        
    sigma : float, optional (default=1.0)
        The width of the Gaussian functions.
        
    Attributes
    ----------
    centers_ : array, shape (n_centers, n_features)
        The centers of the basis functions.
        
    sigma_ : float
        The width of the Gaussian functions.
        
    """
    
    def __init__(self, centers, sigma=1.0):
        self.centers = centers
        self.sigma = sigma
        
    def transform(self, X):
        """Apply the radial basis function encoding to a data array

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data array to be transformed.

        Returns
        -------
        encoding : array, shape (n_samples, n_centers)
            The radial basis function encoding of the input data.
        """
            
        n_samples, n_features = X.shape
        n_centers = self.centers.shape[0]
        encoding = np.zeros((n_samples, n_centers))
        
        for i in range(n_samples):
            for j in range(n_centers):
                diff = X[i] - self.centers[j]
                encoding[i, j] = np.exp(- np.dot(diff, diff) / (2 * self.sigma**2))
                
        return encoding
