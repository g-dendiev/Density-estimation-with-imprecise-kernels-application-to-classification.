from classes.Kernels.Kernel import Kernel
from classes.KernelContext import KernelContext

class TriangularKernel(Kernel):
    """
    Kernel Triangulaire
    """

    # Constructeur de classe
    def __init__(self, h):
        super().__init__(h)     #super() -> pour appeler le parent
        self._name = "Kernel Triangulaire"

    def value(self, centerPoint, seekedPoint):
        if abs(centerPoint - seekedPoint) <= (self.bandwidth / 2):
            # Thales !!!!
            return 2/self._bandwidth - 4*abs(centerPoint - seekedPoint)/(self._bandwidth*self.bandwidth)
        else:
            # Pas dans le Kernel
            return 0

    def kernelFunction(self, u):
        """
            Fonction à définir, associé au kernel. Voir page wikipédia sur les fonctions de kernel
        """
        abs_u = abs(u)
        if abs_u <= 1:
            return 1-abs_u
        return 0

    def testUnitaires():
        """Génération de la multimodale de test (faite à la main)"""
        sampleTestUni = [2, 2.62, 2.84, 2.95, 4.3, 4.7, 4.97, 5.31, 5.7, 7]

        """Déclaration des structures devant contenir le min et le max de h et de f(h)"""
        maxStruct = {
            'potentialHValue': -1,
            'maxedValue': -1
        }

        maxStructFromInt_1 = {
            'potentialHValue': -1,
            'minValue': -1
        }

        minStructFromInt_1 = {
            'potentialHValue': -1,
            'minValue': -1
        }

        maxStructFromInt_2 = {
            'potentialHValue': -1,
            'minValue': -1
        }

        minStructFromInt_2 = {
            'potentialHValue': -1,
            'minValue': -1
        }

        """Déclaraiton du Kernel utilisé """

        TKernel = KernelContext(sampleTestUni, TriangularKernel(0.1), 1)

        """Définition des matrices des résultats"""

        # ComputeHMax
        WaitedHMaxResults_0 = [ [8, .058765625],
                                [6.41, .07717563],
                                [1.24, 0.165842872],
                                [.64, .244140625],
                                [2.626666667, 0.18943737],
                                [.12, .625],
                                [1.98, .14947454],
                                [5.856, .09025235],
                                [7.986666667, .06624668],
                                [10.6, .051966892]]

        # ComputeHMax/MinFromIntervalle avec h=2 et eps = 0.5
        WaitedHMaxResults_1 = [ [2.5, .008],
                                [2.5, .05744],
                                [1.5, 0.15955555],
                                [1.5, .19022222],
                                [2.5, 0.19088],
                                [1.5, .24266666],
                                [1.98, .14947454],
                                [2.5, .08288],
                                [2.5, .0272],
                                [2.5, 0.008] ]

        WaitedHMinResults_1 = [ [1.5, 0],
                                [1.5, .02222222],
                                [2.5, 0.12464],
                                [2.5, .1728],
                                [1.5, 0.161333333],
                                [2.5, .18992],
                                [1.5, .141333333],
                                [1.5, .07555555555],
                                [1.5, .02222222222],
                                [1.5, 0.0] ]

        # ComputeHMax/MinFromIntervalle avec h=3 et eps = 1
        WaitedHMaxResults_2 = [ [4, .184937],
                                [4, .066375],
                                [2, 0.13975],
                                [2, .17775],
                                [2.6266667, 0.18943737],
                                [2, .199],
                                [2, .1495],
                                [3.986666667, .08745708],
                                [4, .049875],
                                [4, 0.0188125] ]

        WaitedHMinResults_2 = [ [2, 0],
                                [2, .03975],
                                [4, 0.1163125],
                                [4, .150625],
                                [4, .1651875],
                                [4, .1648125],
                                [4, .1274375],
                                [2, .07525],
                                [2, .025],
                                [2, 0.0] ]

        for i in range(0,9):
            maxStruct = TKernel.computeHMax(i)
            maxStructFromInt_1 = TKernel.computeHMaxFromInterval(i, 2, 0.5)
            minStructFromInt_1 = TKernel.computeHMinFromInterval(i, 2, 0.5)
            maxStructFromInt_2 = TKernel.computeHMaxFromInterval(i, 3, 1)
            minStructFromInt_2 = TKernel.computeHMinFromInterval(i, 3, 1)
            if (abs(maxStruct['potentialHValue'] - WaitedHMaxResults_0[i][0]) or
                    abs(maxStruct['maxedValue'] - WaitedHMaxResults_0[i][1]) or
                    abs(maxStructFromInt_1['potentialHValue'] - WaitedHMaxResults_1[i][0]) or
                    abs(maxStructFromInt_1['maxedValue'] - WaitedHMaxResults_1[i][1]) or
                    abs(minStructFromInt_1['potentialHValue'] - WaitedHMinResults_1[i][0]) or
                    abs(minStructFromInt_1['minValue'] - WaitedHMinResults_1[i][1]) or
                    abs(maxStructFromInt_2['potentialHValue'] - WaitedHMaxResults_2[i][0]) or
                    abs(maxStructFromInt_2['maxedValue'] - WaitedHMaxResults_2[i][1]) or
                    abs(minStructFromInt_2['potentialHValue'] - WaitedHMinResults_2[i][0]) or
                    abs(minStructFromInt_2['minValue'] - WaitedHMinResults_2[i][1])) > .0001:

                print("ComputeHMAx théorique:", WaitedHMaxResults_0[i][0], WaitedHMaxResults_0[i][1])
                print("ComputeHMAx pratique:", maxStruct['potentialHValue'], maxStruct['maxedValue'])

                print("ComputeHMAxFromInt_1 théorique:", WaitedHMaxResults_1[i][0], WaitedHMaxResults_1[i][1])
                print("ComputeHMAxFromInt_1 pratique:", maxStructFromInt_1['potentialHValue'],
                      maxStructFromInt_1['maxedValue'])

                print("ComputeHMinFromInt_1 théorique:", WaitedHMinResults_1[i][0], WaitedHMinResults_1[i][1])
                print("ComputeHMinFromInt_1 pratique:", minStructFromInt_1['potentialHValue'],
                      minStructFromInt_1['minValue'])

                print("ComputeHMAxFromInt_2 théorique:", WaitedHMaxResults_2[i][0], WaitedHMaxResults_2[i][1])
                print("ComputeHMAxFromInt_2 pratique:", maxStructFromInt_2['potentialHValue'],
                      maxStructFromInt_2['maxedValue'])

                print("ComputeHMAxFromInt_1 théorique:", WaitedHMinResults_2[i][0], WaitedHMinResults_2[i][1])
                print("ComputeHMAxFromInt_1 pratique:", minStructFromInt_2['potentialHValue'],
                      minStructFromInt_2['minValue'])

                raise Exception("Problem between test and implementation")
            else:
                return 0
