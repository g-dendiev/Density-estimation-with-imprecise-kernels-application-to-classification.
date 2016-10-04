import numpy as np

from classes.SampleGenerator.GaussianGenerator import GaussianGenerator

class MultimodalGenerator:

    def __init__(self, listParams=[]):
        self._listParams = []
        if listParams:
            for params in listParams:
                self.addGaussianSample(params)

    def getModelsNumber(self):
        return len(self._params)

    @property
    def listParams(self):
        return self._listParams

    def checkValidParams(self, params):
        if len(params) == 3 and params[0] > 0 and params[2] > 0:
            return True
        else:
            return False

    def addGaussianSample(self, params):
        if self.checkValidParams(params):
            self._listParams.append(params)

    def resetSamples(self):
        self._listParams = []

    def generateNormalSamples(self):
        echantillon = np.array([])
        for params in self._listParams:
            model = GaussianGenerator(params[1], params[2])
            echantillon = np.append(echantillon, model.generateSamples(params[0]))
        return echantillon