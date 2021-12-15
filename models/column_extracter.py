
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnExtracter(BaseEstimator, TransformerMixin):
    # return specific column or columns
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.columns]