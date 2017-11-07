import pandas as pd


class ColumnExtractor(TransformerMixin):
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


class CategoricalEncoder(TransformerMixin):
    # Applies the '1 to N-1' binarizer of categorical encoding.       
    def fit(self, X):
        return self
    
    def transform(self, X):
        return pd.get_dummies(X, drop_first=True)

    def fit_transform(self, X, y='deprecated'):
        return pd.get_dummies(X, drop_first=True)