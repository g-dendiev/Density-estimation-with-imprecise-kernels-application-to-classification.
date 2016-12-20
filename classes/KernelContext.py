import numpy as np
import math

class KernelContext:
    """
        Classe ayant pour but de s'assurer que nos calculs de kernels sont corrects !
    """

    def __init__(self, dataset, kernel, step=0.1):
        self.dataset = dataset
        self.kernel = kernel
        self.step = step
        self.updateDomain()

    def updateDomain(self):
        minDomain = min(self.dataset) - self.kernel.bandwidth
        maxDomain = max(self.dataset) + self.kernel.bandwidth
        nbPoints = math.floor(maxDomain - minDomain) / self.step + 1
        self.domain = np.linspace(minDomain, maxDomain, nbPoints)

    def computeDensityOnPoint(self, kernelCenterPoint):
        fx = 0
        for point in self.dataset:
            fx += self.kernel.kernelFunction((kernelCenterPoint - point)/self.kernel.bandwidth)
        fx = fx / (self.kernel.bandwidth * len(self.dataset))
        return fx

    def computeTotalDensity(self):
        fx = []
        for point in self.domain:
            fx.append(self.computeDensityOnPoint(point))
        return fx, self.domain

    def setStep(self, step):
        self.step = step
        self.updateDomain()

    def setDataset(self, dataset):
        self.dataset = dataset
        self.updateDomain()

    def setBandwidth(self, bandwith):
        self.kernel.bandwidth = bandwith
        self.updateDomain()

    def distanceFromDataset(self, centerPoint):
        distTab = []
        for point in self.dataset:
            dist = abs(point-centerPoint)
            distTab.append(dist)
        return sorted(distTab)

    def computeHMax(self, centerPoint):
        if not(self.kernel):
            raise Exception("No Kernel setted")

        sortedDistances = self.distanceFromDataset(centerPoint)

        # Structure permettant de stocker les infos sur le courant H_sup
        maxStruct = {
            'potentialHValue': -1,
            'maxedValue': -1
        }

        for i in range(len(sortedDistances)):

            nbPointsLocal = i + 1

            # On calcule la somme des distances comprises entre le centre et la distance évaluée et on l'affecte
            hMax = 4 * sum(sortedDistances[0:i + 1]) / nbPointsLocal
            self.setBandwidth(hMax)

            # On va calculer la valeur associée à ce h
            localMax = self.computeDensityOnPoint(centerPoint)

            # On compare pour ne garder que le maximum
            if localMax > maxStruct['maxedValue']:
                maxStruct['maxedValue'] = localMax
                maxStruct['potentialHValue'] = hMax

        return maxStruct

    def computeHMaxFromInterval(self, centerPoint, h, epsi):
        if not(self.kernel):
            raise Exception("No Kernel setted")

        sortedDistances = self.distanceFromDataset(centerPoint)

        # Structure permettant de stocker les infos sur le courant H_sup
        maxStruct = {
            'potentialHValue': -1,
            'maxedValue': -1
        }

        borneInf = (h - epsi)
        self.setBandwidth(borneInf)

        localMaxInf = self.computeDensityOnPoint(centerPoint)

        # On compare pour ne garder que le maximum
        if localMaxInf > maxStruct['maxedValue']:
            maxStruct['maxedValue'] = localMaxInf
            maxStruct['potentialHValue'] = borneInf

        borneSup = (h + epsi)
        self.setBandwidth(borneSup)

        localMaxSup = self.computeDensityOnPoint(centerPoint)

        # On compare pour ne garder que le maximum
        if localMaxSup > maxStruct['maxedValue']:
            maxStruct['maxedValue'] = localMaxSup
            maxStruct['potentialHValue'] = borneSup

        for i in range(len(sortedDistances)):

            nbPointsLocal = i + 1

            # On calcule la somme des distances comprises entre le centre et la distance évaluée et on l'affecte
            hMax = 4 * sum(sortedDistances[0:i + 1]) / nbPointsLocal

            if hMax <= borneInf:
                continue

            if hMax >= borneSup:
                break

            self.setBandwidth(hMax)

            # On va calculer la valeur associée à ce h
            localMax = self.computeDensityOnPoint(centerPoint)

            # On compare pour ne garder que le maximum
            if localMax > maxStruct['maxedValue']:
                maxStruct['maxedValue'] = localMax
                maxStruct['potentialHValue'] = hMax

        return maxStruct

    def computeHMinFromInterval(self, centerPoint, h, epsi):
        if not(self.kernel):
            raise Exception("Pas de kernel")

        sortedDistances = self.distanceFromDataset(centerPoint)

        # Structure permettant de stocker les infos sur le courant H_inf
        minStruct = {
            'potentialHValue': -1,
            'minValue': -1
        }

        borneInf = (h - epsi)
        self.setBandwidth(borneInf)

        localMinInf = self.computeDensityOnPoint(centerPoint)

        # On affecte dans tous les cas, on comparera après !
        minStruct['minValue'] = localMinInf
        minStruct['potentialHValue'] = borneInf

        borneSup = (h + epsi)
        self.setBandwidth(borneSup)

        localMinSup = self.computeDensityOnPoint(centerPoint)

        # On compare pour ne garder que le maximum
        if localMinSup < minStruct['minValue']:
            minStruct['minValue'] = localMinSup
            minStruct['potentialHValue'] = borneSup

        for dist in sortedDistances:

            if dist*2 <= borneInf:
                continue

            if dist*2 >= borneSup:
                break

            hTmp = 2*dist
            self.setBandwidth(hTmp)

            # On va calculer la valeur associée à ce h
            localMin = self.computeDensityOnPoint(centerPoint)

            # On compare pour ne garder que le maximum
            if localMin < minStruct['minValue']:
                minStruct['minValue'] = localMin
                minStruct['potentialHValue'] = hTmp

        return minStruct