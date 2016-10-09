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

plt.plot(defDomain, yOnDomain)
plt.plot(defDomain, yHMinOnDomain)
plt.plot(defDomain, yHMaxOnDomain)
plt.show()