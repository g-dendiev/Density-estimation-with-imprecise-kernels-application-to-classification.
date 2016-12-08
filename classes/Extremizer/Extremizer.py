

class Extremizer:
    """
    Class destinée à calculer des maximum / minimum sur des intervalles
    """

    def __init__(self, dataset=None, centerPoint=None, kernel=None):
        self.dataset = dataset
        self.centerPoint = centerPoint
        self.kernel = kernel
        if self.dataset and self.centerPoint:
            self.distance = self.distanceFromDataset()
        else:
            raise Exception("Aucun dataset fourni")

    def distanceFromDataset(self):
        distTab = []
        for pt in self.dataset:
            dist = abs(pt-self.centerPoint)
            distTab.append(dist)
        return sorted(distTab)

    def computeHMax(self):
        if not(self.kernel):
            raise Exception("No Kernel setted")

        # Structure permettant de stocker les infos sur le courant H_sup
        maxStruct = {
            'potentialHValue': -1,
            'maxedValue': -1
        }

        for i in range(len(self.distance)):

            # BUGFIX ?
            nbPointsLocal = i + 1

            # On calcule la somme des distances comprises entre le centre et la distance évaluée et on l'affecte
            hMax = 4 * sum(self.distance[0:i + 1]) / nbPointsLocal
            self.kernel.bandwidth = hMax

            #print("Voila un résultat : {} avec {} points".format(maxLocal, nbPointsLocal))
            #print("Local sum = {}".format(sum(dist2[0:i + 1])))

            # On va calculer la valeur associée à ce h
            localMax = 0
            for point in self.dataset:
                if abs(point - self.centerPoint) <= hMax:
                    localMax += self.kernel.value(self.centerPoint, point)
            #print("Value du max de f(h) : {}".format(localMax))

            # On compare pour ne garder que le maximum
            if localMax > maxStruct['maxedValue']:
                maxStruct['maxedValue'] = localMax
                maxStruct['potentialHValue'] = hMax

        print("Max de F(h) : {} avec h = {}".format(maxStruct['maxedValue'], maxStruct['potentialHValue']))
        return maxStruct

