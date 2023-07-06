import numpy as np
import pandas as pd

class CategoricalTransformer:
    
    def __init__(self, index_of_col: int):
        self._index_of_col = index_of_col
        self._a = None
        self._b = None
    
    def fit(self, x, y):
        if isinstance(x, pd.DataFrame):
            x = x.values
        self._a = y[x[:, self._index_of_col] == 'A'].mean()
        self._b = y[x[:, self._index_of_col] == 'B'].mean()
    
    def transform(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.values
        x[:, self._index_of_col] = np.where(x[:, self._index_of_col] == 'A', self._a, self._b)
        return x
        
    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x)
