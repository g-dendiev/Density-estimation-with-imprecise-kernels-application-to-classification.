import matplotlib.pylab as plt
import numpy as np
import scipy.integrate as deri

import csv

from classes.Kernels.TriangularKernel import TriangularKernel
from classes.SampleGenerator.MultimodalGenerator import MultimodalGenerator
from classes.Extremizer.Extremizer import Extremizer


def my_range(start, end, step):
    while start <= end:
        yield start
        start += step

def my_range2(start, end, step):
    while start <= end:
        yield start
        start *= step

dataset=[]


with open('resultEpsilon.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['nbPoints'] + ['hOpt'] + ['epsilon'] + ['hInfTri'] + ['hSupTri'] + ['fHInf'] + ['fHSup']+ ['ecartRalatif'] +  ['aireHOpt'] + ['rapport en % de Epsilon sur hOpt'])




"""Boucle sur le nombre de points :"""
for nbPoints in my_range2(1,1050,2):    #Partir avec peu de points (10) et aller jusqu'à 1000. Doubler le nombre à chaque fois.

    """Def du domaine d'étude, on a ici nbPoints de 10 à 1000 points entre -4 et 12"""
    defDomain = np.linspace(-4, 12, nbPoints)

    """Boucle permettant de générer 20 multimodales différentes"""
    for i in range(1,5,1): #on voit pour 5 modales différentes pour l'instant

        """Génération de la multimodale """
        sample = MultimodalGenerator([(100,-1,1),(400,5,2)]).generateNormalSamples()

        """Calcul de l'écart-type de notre multimodale"""

        ecartType = np.std(sample)

        """ Cf fonctions de pierrot pour hMin Opt et hMaxOpt après avoir un hmin et hMax en fonction de epsilon"""

        hOpt = (1.06*ecartType*nbPoints)**(-1/5)

        """Initialisation extremizer"""

        tKernelTri=TriangularKernel(hOpt)

        for j in sample:
            dataset.append(j)

        extremizer=Extremizer(dataset,hOpt,tKernelTri)


        for epsilon in my_range(0.01, hOpt-0.05*hOpt, 0.05*hOpt): # augmenter de 5% de hOpt au fur et à mesure
            """Définition des tableaux de valeurs pour nos tests"""
            yTriHMinOnDomain = []
            yTriHMaxOnDomain = []


            """On trouve le hInf et le hSup (les h donnant  les f(h) suivant : le plus petit et le plus grand"""
            hInfTriDict =extremizer.computeHMinFromInterval(hOpt,epsilon)
            hInfTri=hInfTriDict.get('potentialHValue')
            fHInf=hInfTriDict.get('minValue')
            hSupTriDict =extremizer.computeHMaxFromInterval(hOpt,epsilon)
            hSupTri = hSupTriDict.get('potentialHValue')
            fHSup=hSupTriDict.get('maxedValue')

            aireExacte = deri.trapz(sample)


            ecartRalatif = ((abs(fHSup-fHInf))/fHInf)*100
            rapportEpsHOpt = epsilon/hOpt*100

            with open('resultEpsilon.csv', 'a') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow([nbPoints] + [hOpt] + [epsilon] + [hInfTri] + [hSupTri] + [fHInf] + [fHSup]+ [ecartRalatif] +  [aireExacte] + [rapportEpsHOpt])
