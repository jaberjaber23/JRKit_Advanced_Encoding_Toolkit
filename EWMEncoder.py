import pandas as pd

"""Exponentially Weighted Mean Encoder for both numerical and categorical data.

    Parameters
    ----------
    alpha : float, optional (default=0.1)
        The smoothing factor for the exponentially weighted mean.

    Attributes
    ----------
    alpha : float
        The smoothing factor for the exponentially weighted mean.
    """

class EWMEncoder:
    
    
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def fit_transform(self, df, column, target):
        """Encodes the specified column in the input dataframe `df`.

        The input `df` is transformed in-place and returned. The encoded data is stored in a new column in the dataframe,
        with the name `<column>_ewma`.

        Parameters
        ----------
        df : pandas.DataFrame
            The input dataframe.
        column : str
            The name of the column to encode.
        target : str
            The name of the target column.

        Returns
        -------
        pandas.DataFrame
            The input dataframe with the encoded column added.

        Raises
        ------
        Exception
            If the data type of the specified column is not supported (float64, int64, or object).
        """
        
        if df[column].dtype == 'float64' or df[column].dtype == 'int64':
            result = []
            average = df[target][0]
            df[column + '_ewma'] = 0.0
            for i in range(1, len(df)):
                average = (1-self.alpha) * average + self.alpha * df[target][i]
                result.append(average)
            df[column + '_ewma'] = result
            
            
        elif df[column].dtype == 'object':
            mean_encoded = df.groupby(column)[target].mean()
            df[column + '_ewma'] = df[column].map(mean_encoded)
            
            
        else:
            raise Exception("Data type not supported")
        return df
