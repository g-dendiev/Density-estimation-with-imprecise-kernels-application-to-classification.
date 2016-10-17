import matplotlib.pylab as plt
import numpy as np

from classes.Kernels.TriangularKernel import TriangularKernel
from classes.SampleGenerator.MultimodalGenerator import MultimodalGenerator

hTest = 3
"""valeur qui nous permet d'être proche à 10^-precis près du premier point en dehors de [x-hTest/2 ; x+hTest/2]"""
precis = 6
sample = MultimodalGenerator([(100,-1,1),(400,5,2)]).generateNormalSamples()
tKernel = TriangularKernel(hTest)
defDomain = np.linspace(-4,12,200)

yOnDomain = []
yHMinOnDomain = []
yHMaxOnDomain = []

for x in defDomain:

    fx = 0
    gx = 0
    hx = 0
    hMin=0
    hMax=0

    for j in sample:
        fx += tKernel.value(x,j)

        """On met dans hMin le point le plus loin de x et appartenant à [x-hTest/2 ; x+hTest/2]"""
        if (tKernel.value(x,j)!=0 and abs(x-j)>hMin): hMin=abs(x-j)

        """On met dans hMax la distance entre x et le premier point en dehors de notre intervalle [x-hTest/2 ; x+hTest/2] puis on lui retranche un nombre petit correspondant à 10^-precis, on def precis auparavant"""
        if (tKernel.value(x, j) == 0 and abs(x - j) < hMax): hMax = abs(x - j)
    yOnDomain.append(fx)

    """Ici on passe au Kernel triangulaire avec un h minimal pour le point x dans le domaine de definition"""

    tKernel = TriangularKernel(hMin)

    for j in sample:
        gx += tKernel.value(x,j)
    yHMinOnDomain.append(gx)

    """Ici on passe au Kernel triangulaire avec un h maximal pour le point x dans le domaine de definition
    -> le - 10^-precis evite d'être sur le premier point en dehors de notre interval initial"""

    tKernel = TriangularKernel(hMax-10^-precis)

    for j in sample:
        hx += tKernel.value(x, j)
    yHMaxOnDomain.append(hx)

    """Retour au Kernel avec le hTest défini initialement"""

    tKernel = TriangularKernel(hTest)

plt.figure(figsize=(10,8))

"""Histogramme pour voir la tête de la répartition"""
hist, bins = np.histogram(sample, bins=15)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.subplot(223)
barlist = plt.bar(center, hist, align='center', width=width)
for bar in barlist:
    bar.set_color('y')

plt.plot(defDomain, yOnDomain, label="Brute Force")
plt.plot(defDomain, yHMinOnDomain, label="HMin")
plt.plot(defDomain, yHMaxOnDomain, label="HMax")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.gca().set_position([0, 0, 0.8, 0.8])
plt.show()