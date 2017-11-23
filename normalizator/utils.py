import pandas as pd
import numpy as np
import copy
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



class LimitSetter(BaseEstimator, TransformerMixin):
    # WARNING: Cannot be used within a Pipeline. The current sklearn pipeline
    # does not support subsampling both X and Y datasets.
    
    # Defines the limit range for each variable in the dataset.
    # Used to guarantee the test dataset is between the range of the training set.
    def __init__(self, minimum=[], maximum=[]):
        self.minimum = minimum
        self.maximum = maximum

    def fit(self, X, y='deprecated'):
        self.minimum = np.min(X, axis=0)
        self.maximum = np.max(X, axis=0)
        return self

    def transform(self, X, y='deprecated'):
        X_ = copy.copy(X)
        id_rows = np.all((X_ >= self.minimum) & (X_ <= self.maximum), axis=1)
        self.X_ = X_[id_rows]
        if type(y)!=str:
            y = copy.copy(y)[id_rows]
            return self.X_, y
        else:
            return self.X_

    def fit_transform(self, X, y='deprecated'):
        self.fit(X, y)
        self.transform(X, y)
        if type(y) == str:
            return self.X_
        else:
            return self.X_, y
