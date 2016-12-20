import numpy as np
import math

class KernelContext:
    """
        Classe ayant pour but de s'assurer que nos calculs de kernels sont corrects !
        En passant le dataset et le kernel en argument, on peut restreindre toutes nos opérations dans cette classe !
    """

    def __init__(self, dataset, kernel, step=0.1):
        """
            Constructeur de classe, le param step défini le degré de précision de définition du domaine d'étude
            ex : Si le dataset possède des valeurs comprises entre 1 et 10, la classe va générer un domaine entre ces points.
            Si step = 1, les points générés le seront de 1 en 1 (1, 2, 3..., 9, 10).
            Avec un step par défaut à 0.1, on aura une étude en chaque point (1, 1.1, 1.2, ..., 9.9, 10)
        """
        self.dataset = dataset
        self.kernel = kernel
        self.step = step
        self.updateDomain()


    def updateDomain(self):
        """
            Fonction qui doit être call à chaque modification du dataset, du step ou du bandwidth du kernel,
             afin de reclaculer le domaine de définition de l'étude
        """

        # Le domaine de def est défini par [min(Dataset) - h, max(Dataset) + h]
        minDomain = min(self.dataset) - self.kernel.bandwidth
        maxDomain = max(self.dataset) + self.kernel.bandwidth

        # On gère les valeurs non entière pour le nombre de points
        nbPoints = math.floor(maxDomain - minDomain) / self.step + 1

        # Création du domaine de définition
        self.domain = np.linspace(minDomain, maxDomain, nbPoints)

    def computeDensityOnPoint(self, kernelCenterPoint):
        """
            Fonction qui calcule la densité en 1 point, supposé être le centre du kernel
            Les paramètres du kernel sont ceux du kernel passé en attribut lors de la construction
        """
        fx = 0
        for point in self.dataset:
            fx += self.kernel.kernelFunction((kernelCenterPoint - point)/self.kernel.bandwidth)
        fx = fx / (self.kernel.bandwidth * len(self.dataset))
        return fx

    def computeTotalDensity(self):
        """
            Fonction qui a pour but de calculer l'ensemble de la densité, sur les points déterminés dans self.domain
        """
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
        """
            Fonction utilitaire qui calcule un tableau de distance en fonction du dataset fourni.
            Cela sert dans le cas de nos recherches pour HMax / HMin
        """
        distTab = []
        for point in self.dataset:
            dist = abs(point-centerPoint)
            distTab.append(dist)
        return sorted(distTab)

    def computeHMax(self, centerPoint):
        """
            Fonction qui retourne la valeur maximale atteinte par la fonction de densité en faisant varier h, par rapport ay dataset fourni
        """

        # On s'assure que le kernel est setté
        if not(self.kernel):
            raise Exception("No Kernel setted")

        # On créé notre tableau de distance
        sortedDistances = self.distanceFromDataset(centerPoint)

        # Structure permettant de stocker les infos sur le courant H_sup
        maxStruct = {
            'potentialHValue': -1,
            'maxedValue': -1
        }

        # Ensuite, on commence à itérer pour trouver HMax (en rapport avec la démo dans notre document)
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
        """
            Fonction qui retourne la valeur maximale atteinte par la fonction de densité en faisant varier h,
            par rapport au dataset fourni, dans un interval donné par [h-epsi, h+epsi]
        """

        # On s'assure que le kernel est setté
        if not(self.kernel):
            raise Exception("No Kernel setted")

        # Tableau de distance trié
        sortedDistances = self.distanceFromDataset(centerPoint)

        # Structure permettant de stocker les infos sur le courant H_sup
        maxStruct = {
            'potentialHValue': -1,
            'maxedValue': -1
        }

        # On set pour la borne inf
        borneInf = (h - epsi)
        self.setBandwidth(borneInf)

        localMaxInf = self.computeDensityOnPoint(centerPoint)

        # On compare pour ne garder que le maximum
        if localMaxInf > maxStruct['maxedValue']:
            maxStruct['maxedValue'] = localMaxInf
            maxStruct['potentialHValue'] = borneInf

        # On set pour la borne sup
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
        """
            Fonction qui retourne la valeur maximale atteinte par la fonction de densité en faisant varier h,
            par rapport au dataset fourni, dans un interval donné par [h-epsi, h+epsi]
        """

        # On s'assure que le kernel est setté
        if not(self.kernel):
            raise Exception("Pas de kernel")

        # Tableau de distance trié
        sortedDistances = self.distanceFromDataset(centerPoint)

        # Structure permettant de stocker les infos sur le courant H_inf
        minStruct = {
            'potentialHValue': -1,
            'minValue': -1
        }

        # On commence par calculer les valeurs en borne inf
        borneInf = (h - epsi)
        self.setBandwidth(borneInf)

        localMinInf = self.computeDensityOnPoint(centerPoint)

        # On affecte dans tous les cas, on comparera après !
        minStruct['minValue'] = localMinInf
        minStruct['potentialHValue'] = borneInf

        # Puis on calcule les valeurs en borne sup
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