import numpy as np

class GaussianGenerator:

    def __init__(self, mu=0, sd=1):
        self._mu = mu
        self._sd = sd

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, mu):
        self._mu = mu

    @property
    def sd(self):
        return self._sd

    @sd.setter
    def sd(self, sd):
        self._sd = sd

    def generateSamples(self, n = 10):
        return np.random.normal(self.mu, self.sd, n)

