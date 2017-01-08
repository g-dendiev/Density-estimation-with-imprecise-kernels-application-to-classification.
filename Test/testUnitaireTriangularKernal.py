import matplotlib.pylab as plt
import numpy as np

from statistics import stdev


from classes.Kernels.TriangularKernel import TriangularKernel
from classes.KernelContext import KernelContext
from classes.SampleGenerator.MultimodalGenerator import MultimodalGenerator


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


"""Déclaraiton du Kernel utilisé """

TKernel = KernelContext(sampleTestUni, TriangularKernel(0.1), 1)

"""Définition des matrices des résultats"""

# ComputeHMax
WaitedHMaxResults_0 = [[8.478, .058976174],
                       [6.478, .077184316],
                       [1.205, 0.16975104],
                       [.1, .5],
                       [2.348888889, 0.19157943],
                       [.06, .8333333333],
                       [2.006666666666, .149501661],
                       [5.522, .090546903],
                       [7.522, .066471683],
                       [9.522, .052509977]]

# ComputeHMax/MinFromIntervalle avec h=2 et eps = 0.5
WaitedHMaxResults_1 = [[2.5, .008],
                       [2.5, .05744],
                       [1.5, 0.15955555],
                       [1.5, .2048888889],
                       [2.348888889, 0.19157943],
                       [1.5, .24266666],
                       [2.006666666666, .149501661],
                       [2.5, .08288],
                       [2.5, .0272],
                       [2.5, 0.008]]

WaitedHMinResults_1 = [[1.5, 0],
                       [1.5, .02222222],
                       [2.5, 0.12464],
                       [2.5, .17808],
                       [1.5, 0.161333333],
                       [2.5, .18992],
                       [1.5, .141333333],
                       [1.99333333, .07525083612040134],
                       [1.5, .02222222222],
                       [1.5, 0.0]]

for i in range(0, 10):
    maxStruct = TKernel.computeHMax(i)
    maxStructFromInt_1 = TKernel.computeHMaxFromInterval(i, 2, 0.5)
    minStructFromInt_1 = TKernel.computeHMinFromInterval(i, 2, 0.5)
    if (abs(maxStruct['potentialHValue'] - WaitedHMaxResults_0[i][0]) or
            abs(maxStruct['maxedValue'] - WaitedHMaxResults_0[i][1]) or
            abs(maxStructFromInt_1['potentialHValue'] - WaitedHMaxResults_1[i][0]) or
            abs(maxStructFromInt_1['maxedValue'] - WaitedHMaxResults_1[i][1]) or
            abs(minStructFromInt_1['potentialHValue'] - WaitedHMinResults_1[i][0]) or
            abs(minStructFromInt_1['minValue'] - WaitedHMinResults_1[i][1]) or
            abs(maxStructFromInt_2['potentialHValue'] - WaitedHMaxResults_2[i][0]) or
            abs(maxStructFromInt_2['maxedValue'] - WaitedHMaxResults_2[i][1]) or
            abs(minStructFromInt_2['potentialHValue'] - WaitedHMinResults_2[i][0]) or
            abs(minStructFromInt_2['minValue'] - WaitedHMinResults_2[i][1])) > .01:

        print("Problème, voir suite :")

        print("étape",i)
        print("ComputeHMAx théorique:", WaitedHMaxResults_0[i][0], WaitedHMaxResults_0[i][1])
        print("ComputeHMAx pratique:", maxStruct['potentialHValue'], maxStruct['maxedValue'])

        print("ComputeHMAxFromInt_1 théorique:", WaitedHMaxResults_1[i][0], WaitedHMaxResults_1[i][1])
        print("ComputeHMAxFromInt_1 pratique:", maxStructFromInt_1['potentialHValue'],
              maxStructFromInt_1['maxedValue'])

        print("ComputeHMinFromInt_1 théorique:", WaitedHMinResults_1[i][0], WaitedHMinResults_1[i][1])
        print("ComputeHMinFromInt_1 pratique:", minStructFromInt_1['potentialHValue'],
              minStructFromInt_1['minValue'])



        raise Exception("Problem between test and implementation")
    else:
        print("etape:",i,"tout est ok :)")


