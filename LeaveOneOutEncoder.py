import pandas as pd
import numpy as np


"""Perform Leave-One-Out encoding on a categorical feature.
    
    Leave-One-Out Encoding is a technique for encoding categorical features. 
    It replaces each category value with the mean target value of all the samples 
    that belong to that category, excluding the sample being processed.
    
    Parameters
    ----------
    cols : list, optional (default=None)
        A list of column names to be encoded. If None, all columns in the input 
        data will be used.
    
    target : string, optional (default=None)
        The target column name. This column must exist in the input data and will 
        be used to calculate the mean target value for each category.
        
    return_df : bool, optional (default=False)
        Whether to return the result as a pandas DataFrame. If False, the result 
        will be returned as a numpy array.
        
    Attributes
    ----------
    means_ : dict
        The mean target value for each category in each column.
        
    """

class LeaveOneOutEncoder:
     
    def __init__(self, cols=None, target=None, return_df=False):
        self.cols = cols
        self.target = target
        self.return_df = return_df
        
    def fit(self, X, y):
        """Fit the Leave-One-Out encoder to the input data.
        
        Parameters
        ----------
        X : pandas DataFrame
            The input feature data.
            
        y : pandas Series, 1d numpy array
            The target data.
            
        Returns
        -------
        self : object
            Returns the instance of the LeaveOneOutEncoder.
            
        """
        # Store the original data for later use
        self.X_ = X
        if self.cols is None:
            self.cols = X.columns
        
        if y.name not in self.X_.columns:
            raise KeyError(f"Target column {y.name} not found in X data")
        
        self.means_ = {}
        for col in self.cols:
            self.means_[col] = self.X_.groupby(col)[y.name].mean()
        
        return self
    
    
    def transform(self, X, y=None):
        """Transform the input data using the Leave-One-Out encoder.
        
        Parameters
        ----------
        X : pandas DataFrame
            The input feature data.
            
        y : pandas Series, 1d numpy array, optional (default=None)
            The target data. If not provided, the target column specified during 
            initialization will be used.
            
        Returns
        -------
        X_transformed : pandas DataFrame or numpy array
            The transformed feature data, returned as a pandas DataFrame or 
            numpy array depending on the return_df parameter.
            
        """
     # Convert input data to pandas dataframe if it is not
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if y is not None and not isinstance(y, (pd.Series, np.ndarray)):
            y = pd.Series(y)

    # Determine target values if not provided
        if y is None:
            if self.target is None:
                y = X.iloc[:, -1]
                X = X.iloc[:, :-1]
            else:
                y = X[self.target]
                X = X.drop(columns=self.target)

        # Transform data
        X_transformed = X.copy()
        for col in self.cols:
            X_transformed[col] = X_transformed[col].map(self.means_[col])
    
        # Return data as pandas dataframe or numpy array
        if self.return_df:
            return X_transformed
        else:
            return X_transformed.values
        
    def fit_transform(self, X, y=None):
        """
        A method that fits the LeaveOneOutEncoder to the data and then transforms the data in one step.
    
        Parameters
        ----------
        X : pandas DataFrame, shape (n_samples, n_features)
        The training input samples.
    y : pandas Series, shape (n_samples,) or None (default=None)
        The target values. If None, the last column of X is used as the target.

    Returns
    -------
    X_transformed : pandas DataFrame, shape (n_samples, n_features)
        The transformed data, where each categorical feature is replaced by the mean target value.
    """
        self.fit(X, y)
        return self.transform(X, y)