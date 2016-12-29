

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
        return sorted(distTab) #retourne le tableau des distances triées

    def computeHMax(self):
        if not(self.kernel):
            raise Exception("No Kernel setted")

        # Structure permettant de stocker les infos sur le courant H_sup
        maxStruct = {
            'potentialHValue': -1,
            'maxedValue': -1
        }

        for i in range(len(self.distance)):

            nbPointsLocal = i + 1

            # On calcule la somme des distances comprises entre le centre et la distance évaluée et on l'affecte
            hMax = 4 * sum(self.distance[0:i + 1]) / nbPointsLocal
            self.kernel.bandwidth = hMax

            # On va calculer la valeur associée à ce h
            localMax = 0
            for point in self.dataset:
                if abs(point - self.centerPoint) <= hMax:
                    localMax += self.kernel.value(self.centerPoint, point)

            # On compare pour ne garder que le maximum
            if localMax > maxStruct['maxedValue']:
                maxStruct['maxedValue'] = localMax
                maxStruct['potentialHValue'] = hMax

        return maxStruct

    def computeHMaxFromInterval(self, h, epsi):
        if not(self.kernel):
            raise Exception("No Kernel setted")

        # Structure permettant de stocker les infos sur le courant H_sup
        maxStruct = {
            'potentialHValue': -1,
            'maxedValue': -1
        }

        borneInf = (h - epsi)

        self.kernel.bandwidth = borneInf

        localMaxInf = 0
        for point in self.dataset:
            if abs(point - self.centerPoint) <= borneInf:
                localMaxInf += self.kernel.value(self.centerPoint, point)

        # On compare pour ne garder que le maximum
        if localMaxInf > maxStruct['maxedValue']:
            maxStruct['maxedValue'] = localMaxInf
            maxStruct['potentialHValue'] = borneInf

        borneSup = (h + epsi)

        self.kernel.bandwidth = borneSup

        localMaxSup = 0
        for point in self.dataset:
            if abs(point - self.centerPoint) <= borneSup:
                localMaxSup += self.kernel.value(self.centerPoint, point)

        # On compare pour ne garder que le maximum
        if localMaxSup > maxStruct['maxedValue']:
            maxStruct['maxedValue'] = localMaxSup
            maxStruct['potentialHValue'] = borneSup

        for i in range(len(self.distance)):

            nbPointsLocal = i + 1

            # On calcule la somme des distances comprises entre le centre et la distance évaluée et on l'affecte
            hMax = 4 * sum(self.distance[0:i + 1]) / nbPointsLocal

            if hMax <= borneInf:
                continue

            if hMax >= borneSup:
                break

            self.kernel.bandwidth = hMax

            # On va calculer la valeur associée à ce h
            localMax = 0
            for point in self.dataset:
                if abs(point - self.centerPoint) <= hMax:
                    localMax += self.kernel.value(self.centerPoint, point)

            # On compare pour ne garder que le maximum
            if localMax > maxStruct['maxedValue']:
                maxStruct['maxedValue'] = localMax
                maxStruct['potentialHValue'] = hMax

        return maxStruct

    def computeHMinFromInterval(self, h, epsi):
        if not(self.kernel):
            raise Exception("Pas de kernel")

        # Structure permettant de stocker les infos sur le courant H_inf
        minStruct = {
            'potentialHValue': -1,
            'minValue': -1
        }

        borneInf = (h - epsi)

        self.kernel.bandwidth = borneInf

        localMinInf = 0
        for point in self.dataset:
            if abs(point - self.centerPoint) <= borneInf:
                localMinInf += self.kernel.value(self.centerPoint, point)

        # On compare pour ne garder que le maximum
        minStruct['minValue'] = localMinInf
        minStruct['potentialHValue'] = borneInf

        borneSup = (h + epsi)

        self.kernel.bandwidth = borneSup

        localMinSup = 0
        for point in self.dataset:
            if abs(point - self.centerPoint) <= borneSup:
                localMinSup += self.kernel.value(self.centerPoint, point)

        # On compare pour ne garder que le maximum
        if localMinSup < minStruct['minValue']:
            minStruct['minValue'] = localMinSup
            minStruct['potentialHValue'] = borneSup

        for dist in self.distance:

            if dist*2 <= borneInf:
                continue

            if dist*2 >= borneSup:
                break

            hTmp = 2*dist
            self.kernel.bandwidth = hTmp

            # On va calculer la valeur associée à ce h
            localMin = 0
            for point in self.dataset:
                if abs(point - self.centerPoint) <= hTmp:
                    localMin += self.kernel.value(self.centerPoint, point)

            # On compare pour ne garder que le maximum
            if localMin < minStruct['minValue']:
                minStruct['minValue'] = localMin
                minStruct['potentialHValue'] = hTmp

        return minStruct