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
    spamwriter.writerow(['nbPoints'] + ['hOpt'] + ['epsilon'] + ['hInfTri'] + ['hSupTri'] + ['fHInf'] + ['fHSup']+ ['ecartRalatif'] + ['rapport en % de Epsilon sur hOpt'])




"""Boucle sur le nombre de points :"""
for nbPoints in my_range2(10,1000,2):    #Partir avec peu de points (10) et aller jusqu'à 1000. Doubler le nombre à chaque fois.

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

            # for x in defDomain:
            #
            #     """remise à zéro de nos variables pour remplir les tableaux de valeurs de f(hMin) et f(hMax)"""
            #     fx = 0
            #     gx = 0
            #     hx = 0
            #
            #
            #
            #
            #     """Ici on passe aux Kernel avec un h minimal pour le point x dans le domaine de definition"""
            #
            #     tKernelTri = TriangularKernel(hInfTri)
            #
            #
            #     for j in sample:
            #         gx += tKernelTri.value(x,j)
            #     yTriHMinOnDomain.append(gx)
            #
            #
            #     """Ici on passe aux Kernel avec un h maximal pour le point x dans le domaine de definition"""
            #
            #     tKernelTri = TriangularKernel(hSupTri)
            #
            #     for j in sample:
            #         hx += tKernelTri.value(x, j)
            #     yTriHMaxOnDomain.append(hx)
            #
            # """plt.figure(figsize=(10,8))"""
            #
            # """Histogramme pour voir la tête de la répartition"""
            # """hist, bins = np.histogram(sample, bins=15)
            # width = 0.7 * (bins[1] - bins[0])
            # center = (bins[:-1] + bins[1:]) / 2
            # plt.subplot(223)
            # barlist = plt.bar(center, hist, align='center', width=width)
            # for bar in barlist:
            #     bar.set_color('y')"""
            #
            # """ plt.plot(defDomain, yTriHMinOnDomain, label="TriHMin")
            # plt.plot(defDomain, yTriHMaxOnDomain, label="TriHMax")
            # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            # plt.gca().set_position([0, 0, 0.8, 0.8])
            # plt.show()"""
            #
            # aireHMax = deri.trapz(yTriHMaxOnDomain)
            # aireHMin = deri.trapz(yTriHMinOnDomain)
            # aireExacte = deri.trapz(sample)
            #
            # ecartExacEtMin = (abs(aireHMin-aireExacte)/aireExacte)*100
            # ecartExacEtMax = (abs(aireHMax-aireExacte)/aireExacte)*100
            #
            ecartRalatif = ((abs(fHSup-fHInf))/fHInf)*100
            rapportEpsHOpt = epsilon/hOpt*100

            with open('resultEpsilon.csv', 'a') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow([nbPoints] + [hOpt] + [epsilon] + [hInfTri] + [hSupTri] + [fHInf] + [fHSup]+ [ecartRalatif] + [rapportEpsHOpt])
