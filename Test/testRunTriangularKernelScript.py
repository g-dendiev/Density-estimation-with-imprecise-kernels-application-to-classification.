import matplotlib.pylab as plt
import numpy as np

from classes.Kernels.TriangularKernel import TriangularKernel
from classes.SampleGenerator.MultimodalGenerator import MultimodalGenerator


"""Génération de la multimodale """
sample = MultimodalGenerator([(100,-1,1),(400,5,2)]).generateNormalSamples()

"""Nombre de points :"""
nbPoints = 100

"""Def du domaine d'étude, on a ici 200 points entre -4 et 12"""
defDomain = np.linspace(-4,12,nbPoints)


"""Calcul de l'écart-type de notre multimodale"""

ecartType = np.std(sample)

""" Cf readme pour les valeurs de hMin et hMax"""

hOpt = (1.06*ecartType*nbPoints)**(-1/5)

epsilon = 0.1

hMinTri = hOpt - epsilon
hMaxTri = hOpt + epsilon

print("hOpt :", hOpt, "\nepsilon :", epsilon,"\nhMin :",hMinTri, "\nhMax :",hMaxTri)

"""Définition des tableaux de valeurs pour nos tests"""
yTriHMinOnDomain = []
yTriHMaxOnDomain = []


for x in defDomain:

    """remise à zéro de nos variables pour remplir les tableaux de valeurs de f(hMin) et f(hMax)"""
    fx = 0
    gx = 0
    hx = 0



    """Ici on passe aux Kernel avec un h minimal pour le point x dans le domaine de definition"""

    tKernelTri = TriangularKernel(hMinTri)


    for j in sample:
        gx += tKernelTri.value(x,j)
    yTriHMinOnDomain.append(gx)


    """Ici on passe aux Kernel avec un h maximal pour le point x dans le domaine de definition"""

    tKernelTri = TriangularKernel(hMaxTri)

    for j in sample:
        hx += tKernelTri.value(x, j)
    yTriHMaxOnDomain.append(hx)

plt.figure(figsize=(10,8))

"""Histogramme pour voir la tête de la répartition"""
hist, bins = np.histogram(sample, bins=15)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.subplot(223)
barlist = plt.bar(center, hist, align='center', width=width)
for bar in barlist:
    bar.set_color('y')

plt.plot(defDomain, yTriHMinOnDomain, label="TriHMin")
plt.plot(defDomain, yTriHMaxOnDomain, label="TriHMax")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.gca().set_position([0, 0, 0.8, 0.8])
plt.show()