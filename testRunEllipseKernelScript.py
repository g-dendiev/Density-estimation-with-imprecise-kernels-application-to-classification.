import matplotlib.pylab as plt
import numpy as np

from statistics import stdev
from statistics import variance

from classes.Kernels.EllipseKernel import EllipseKernel
from classes.SampleGenerator.MultimodalGenerator import MultimodalGenerator

from math import sqrt
from math import pi

hTestEll =  2*sqrt(pi)
"""valeur qui nous permet d'être proche à 10^-precis près du premier point en dehors de [x-hTest/2 ; x+hTest/2]"""
precis = 6
sample = MultimodalGenerator([(100,-1,1),(400,5,2)]).generateNormalSamples()
tKernelEll = EllipseKernel(hTestEll)
defDomain = np.linspace(-4,12,200)
yEllOnDomain = []
yEllHMinOnDomain = []
yEllHMaxOnDomain = []
ecartH=[]

for x in defDomain:

    ax = 0
    bx = 0
    cx = 0
    hMinEll=0
    hMaxEll=2*hTestEll

    for j in sample:
        ax += tKernelEll.value(x,j)

        """On met dans les hMin les point les plus loin de x et appartenant à [x-hTest/2 ; x+hTest/2]"""
        if (tKernelEll.value(j, x) != 0 and abs(x - j) > hMinEll/2): hMinEll = 2*abs(x - j)

        """On met dans les hMax la distance entre x et le premier point en dehors de notre intervalle [x-hTest/2 ; x+hTest/2] puis on lui retranche un nombre petit correspondant à 10^-precis, on def precis auparavant"""
        if (tKernelEll.value(j, x) == 0 and abs(x - j) < hMaxEll/2): hMaxEll = 2*abs(x - j)
    yEllOnDomain.append(ax)
    print(str(hMinEll) + " / " + str(hMaxEll))

    """Ici on passe aux Kernel avec un h minimal pour le point x dans le domaine de definition"""

    tKernelEll = EllipseKernel(hMinEll)

    for j in sample:
        bx += tKernelEll.value(x,j)
    yEllHMinOnDomain.append(bx)

    """Ici on passe aux Kernel avec un h maximal pour le point x dans le domaine de definition
    -> le - 10^-precis evite d'être sur le premier point en dehors de notre interval initial"""

    hMaxEll -= 10 ^ -precis

    ecart = hMaxEll - hMinEll

    ecartH.append(ecart)

    tKernelEll = EllipseKernel(hMaxEll)

    for j in sample:
        cx += tKernelEll.value(x, j)
    yEllHMaxOnDomain.append(cx)

    """Retour aux Kernel avec les hTest défini initialement"""

    tKernelEll = EllipseKernel(hTestEll)

plt.figure(figsize=(10,8))

"""Histogramme pour voir la tête de la répartition"""
hist, bins = np.histogram(sample, bins=15)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.subplot(223)
barlist = plt.bar(center, hist, align='center', width=width)
for bar in barlist:
    bar.set_color('y')


print(ecartH)
print(" moyenne ecart hMax - hMin :   " + str(stdev(ecartH)) + "    variance : " + str(variance(ecartH)))

plt.plot(defDomain, yEllOnDomain, label="Brute Force Ell")
plt.plot(defDomain, yEllHMinOnDomain, label="EllHMin")
plt.plot(defDomain, yEllHMaxOnDomain, label="EllHMax")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.gca().set_position([0, 0, 0.8, 0.8])
plt.show()