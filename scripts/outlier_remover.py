import numpy as np

class OutlierRemover:
    
    def __init__(self):
        self._lower_lims = {}
        self._upper_lims = {}
    
    def fit(self, x: np.ndarray, y: np.ndarray):
        for i in range(x.shape[1]):
            data = x[:, i].copy()
            mean = data.mean()
            std = data.std()
            self._lower_lims[i] = mean - 6*std
            self._upper_lims[i] = mean + 6*std
    
    def transform(self, x: np.ndarray):
        for i in range(x.shape[1]):
            x[:, i] = np.where(((x[:, i] > self._upper_lims[i]) | (x[:, i] < self._lower_lims[i])), np.nan, x[:, i])
        return x
        
    def fit_transform(self, x: np.ndarray, y: np.ndarray):
        self.fit(x, y)
        return self.transform(x)
