import matplotlib.pylab as plt
import numpy as np

from statistics import stdev
from statistics import variance

from classes.Kernels.TriangularKernel import TriangularKernel
from classes.SampleGenerator.MultimodalGenerator import MultimodalGenerator

hTestTri = 2
"""valeur qui nous permet d'être proche à 10^-precis près du premier point en dehors de [x-hTest/2 ; x+hTest/2]"""
precis = 6
sample = MultimodalGenerator([(100,-1,1),(400,5,2)]).generateNormalSamples()
tKernelTri = TriangularKernel(hTestTri)
defDomain = np.linspace(-4,12,200)
yTriOnDomain = []
yTriHMinOnDomain = []
yTriHMaxOnDomain = []
ecartH=[]


for x in defDomain:

    fx = 0
    gx = 0
    hx = 0
    hMinTri=0
    hMaxTri=2*hTestTri
    ecart=0


    for j in sample:
        fx += tKernelTri.value(x,j)

        """On met dans les hMin les point les plus loin de x et appartenant à [x-hTest/2 ; x+hTest/2]"""
        if (tKernelTri.value(j, x)!=0 and abs(x-j)>hMinTri/2): hMinTri=2*abs(x-j)

        """On met dans les hMax la distance entre x et le premier point en dehors de notre intervalle [x-hTest/2 ; x+hTest/2] puis on lui retranche un nombre petit correspondant à 10^-precis, on def precis auparavant"""
        if (tKernelTri.value(j, x) == 0 and abs(x - j) < hMaxTri/2): hMaxTri = 2*abs(x - j)


    yTriOnDomain.append(fx)
    #print(str(hMinTri) + " / " + str(hMaxTri))
    #ecart = hMaxTri - hMinTri



    """Ici on passe aux Kernel avec un h minimal pour le point x dans le domaine de definition"""

    tKernelTri = TriangularKernel(hMinTri)

    for j in sample:
        gx += tKernelTri.value(x,j)
    yTriHMinOnDomain.append(gx)

    """Ici on passe aux Kernel avec un h maximal pour le point x dans le domaine de definition
    -> le - 10^-precis evite d'être sur le premier point en dehors de notre interval initial"""

    hMaxTri -= 0.0000001

    ecart = hMaxTri - hMinTri

    ecartH.append(ecart)

    tKernelTri = TriangularKernel(hMaxTri)

    for j in sample:
        hx += tKernelTri.value(x, j)
    yTriHMaxOnDomain.append(hx)

    """Retour aux Kernel avec les hTest défini initialement"""

    tKernelTri = TriangularKernel(hTestTri)

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

plt.plot(defDomain, yTriOnDomain, label="Brute Force Tri")
plt.plot(defDomain, yTriHMinOnDomain, label="TriHMin")
plt.plot(defDomain, yTriHMaxOnDomain, label="TriHMax")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.gca().set_position([0, 0, 0.8, 0.8])
plt.show()