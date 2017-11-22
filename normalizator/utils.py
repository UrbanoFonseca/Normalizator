import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

class ColumnExtractor(BaseEstimator, TransformerMixin):
	# To be applied with Pipeline
	# Extracts the list of columns provided on __init__
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X[self.columns]
    
    def fit_transform(self, X, y=None):
        return X[self.columns]


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    # Applies the '1 to N-1' binarizer of categorical encoding.
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X):
        return self
    
    def transform(self, X):
        return pd.get_dummies(X, columns=self.columns, drop_first=True)

    def fit_transform(self, X, y='deprecated'):
        return pd.get_dummies(X, columns=self.columns, drop_first=True)


class ConstantEraser(BaseEstimator, TransformerMixin):
    # Erases constant variables by calculating their variance and
    # erasing if below a certain treshold (1E-20)
    def __init__(self):
        return

    def fit(self, X):
        return self

    def transform(self, X, y='deprecated'):
        return X[:, np.var(X, axis=0)>1E-20]

    def fit_transform(self, X, y='deprecated'):
        return X[:, np.var(X, axis=0)>1E-20]
