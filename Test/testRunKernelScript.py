import matplotlib.pylab as plt
import numpy as np

from classes.Kernels.TriangularKernel import TriangularKernel
from classes.Kernels.EllipseKernel import EllipseKernel
from classes.SampleGenerator.MultimodalGenerator import MultimodalGenerator
from classes.KernelContext import KernelContext

from math import sqrt
from math import pi

hTestTri = 3
hTestEll =  2*sqrt(pi)
"""valeur qui nous permet d'être proche à 10^-precis près du premier point en dehors de [x-hTest/2 ; x+hTest/2]"""
precis = 6
sample = MultimodalGenerator([(100,-1,1),(400,5,2)]).generateNormalSamples()
tKernelTri = TriangularKernel(hTestTri)
tKernelEll = EllipseKernel(hTestEll)
defDomain = np.linspace(-4,12,200)
yTriOnDomain = []
yTriHMinOnDomain = []
yTriHMaxOnDomain = []
yEllOnDomain = []
yEllHMinOnDomain = []
yEllHMaxOnDomain = []

"""
    TEST CODE PIERRE
"""

kc = KernelContext(sample, tKernelTri)
ftest, domain_test = kc.computeTotalDensity()
maxstru = kc.computeHMax(5)
print(maxstru)
kc.setBandwidth(maxstru['potentialHValue'])
ftest2, domain_test2 = kc.computeTotalDensity()


"""
    FIN DU TEST
"""
for x in defDomain:

    fx = 0
    gx = 0
    hx = 0
    hMinTri=0
    hMaxTri=0

    ax = 0
    bx = 0
    cx = 0
    hMinEll=0
    hMaxEll=0

    for j in sample:
        fx += tKernelTri.value(x,j)
        ax += tKernelEll.value(x,j)

        """On met dans les hMin les point les plus loin de x et appartenant à [x-hTest/2 ; x+hTest/2]"""
        if (tKernelTri.value(x,j)!=0 and abs(x-j)>hMinTri): hMinTri=abs(x-j)
        if (tKernelEll.value(x, j) != 0 and abs(x - j) > hMinEll): hMinEll = abs(x - j)

        """On met dans les hMax la distance entre x et le premier point en dehors de notre intervalle [x-hTest/2 ; x+hTest/2] puis on lui retranche un nombre petit correspondant à 10^-precis, on def precis auparavant"""
        if (tKernelTri.value(x, j) == 0 and abs(x - j) < hMaxTri): hMaxTri = abs(x - j)
        if (tKernelEll.value(x, j) == 0 and abs(x - j) < hMaxEll): hMaxEll = abs(x - j)
    yTriOnDomain.append(fx)
    yEllOnDomain.append(ax)

    """Ici on passe aux Kernel avec un h minimal pour le point x dans le domaine de definition"""

    tKernelTri = TriangularKernel(hMinTri)
    tKernelEll = EllipseKernel(hMinEll)

    for j in sample:
        gx += tKernelTri.value(x,j)
        bx += tKernelEll.value(x,j)
    yTriHMinOnDomain.append(gx)
    yEllHMinOnDomain.append(bx)

    """Ici on passe aux Kernel avec un h maximal pour le point x dans le domaine de definition
    -> le - 10^-precis evite d'être sur le premier point en dehors de notre interval initial"""

    tKernelTri = TriangularKernel(hMaxTri-10^-precis)
    tKernelEll = EllipseKernel(hMaxEll - 10 ^ -precis)

    for j in sample:
        hx += tKernelTri.value(x, j)
        cx += tKernelEll.value(x, j)
    yTriHMaxOnDomain.append(hx)
    yEllHMaxOnDomain.append(cx)

    """Retour aux Kernel avec les hTest défini initialement"""

    tKernelTri = TriangularKernel(hTestTri)
    tKernelEll = EllipseKernel(hTestEll)

plt.figure(figsize=(10,8))

"""Histogramme pour voir la tête de la répartition"""
hist, bins = np.histogram(sample, bins=1500)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.subplot(223)
#barlist = plt.bar(center, hist, align='center', width=width)
#for bar in barlist:
#    bar.set_color('y')

plt.plot(domain_test, ftest, label="Avant modif")
plt.plot(domain_test2, ftest2, label="Après modif")
#plt.plot(defDomain, yTriHMinOnDomain, label="TriHMin")
#plt.plot(defDomain, yTriHMaxOnDomain, label="TriHMax")
#plt.plot(defDomain, yEllOnDomain, label="Brute Force Ell")
#plt.plot(defDomain, yEllHMinOnDomain, label="EllHMin")
#plt.plot(defDomain, yEllHMaxOnDomain, label="EllHMax")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.gca().set_position([0, 0, 0.8, 0.8])
plt.show()