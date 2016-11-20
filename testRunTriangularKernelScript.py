import matplotlib.pylab as plt
import numpy as np

from statistics import stdev
from statistics import variance

from classes.Kernels.TriangularKernel import TriangularKernel
from classes.SampleGenerator.MultimodalGenerator import MultimodalGenerator

hMinTri = 0.01
hMaxTri = 100
"""valeurs de départ telles que hOpt appartient à [hMinTri,hMaxTri]"""
epsilonH = 0.1 #ici c'est juste pour tester la valeur de epsilon est arbitraire
"""pointsSuccessifs = 3 #test pour les max et min des f(hMin) et f(hMax) pour voir si on doit ajouter/soustraire epsilon"""
sample = MultimodalGenerator([(100,-1,1),(400,5,2)]).generateNormalSamples()
#tKernelTri = TriangularKernel(hTestTri)
defDomain = np.linspace(-4,12,200)
yTriOnDomain = []
yTriHMinOnDomain = []
yTriHMaxOnDomain = []
ecartH=[]
borne = 12
compteur = 0
compteurZero = 0

"""Var pour avoir l'écart entre 2 points successifs"""
gx_moins_1 = 0
hx_moins_1 = 0


for x in defDomain:

    if (compteur >= 1):
        # on sauv les anciennes vameur de gx et hx
        gx_moins_1 = gx
        hx_moins_1 = hx

    fx = 0
    gx = 0
    hx = 0
    compteur += 1

    """
    for j in sample:
        fx += tKernelTri.value(x,j)
        #On met dans les hMin les point les plus loin de x et appartenant à [x-hTest/2 ; x+hTest/2]
        if (tKernelTri.value(j, x)!=0 and abs(x-j)>hMinTri/2): hMinTri=2*abs(x-j)

       # On met dans les hMax la distance entre x et le premier point en dehors de notre intervalle [x-hTest/2 ; x+hTest/2] puis on lui retranche un nombre petit correspondant à 10^-precis, on def precis auparavant
        if (tKernelTri.value(j, x) == 0 and abs(x - j) < hMaxTri/2): hMaxTri = 2*abs(x - j)
    """

    #yTriOnDomain.append(fx)
    #print(str(hMinTri) + " / " + str(hMaxTri))
    #ecart = hMaxTri - hMinTri



    """Ici on passe aux Kernel avec un h minimal pour le point x dans le domaine de definition"""

    tKernelTri = TriangularKernel(hMinTri)


    for j in sample:
        gx += tKernelTri.value(x,j)
    yTriHMinOnDomain.append(gx)

    if ((gx - gx_moins_1) > borne):
        hMinTri += epsilonH
        print("on a changé h min")




    """Ici on passe aux Kernel avec un h maximal pour le point x dans le domaine de definition
    -> le - 10^-precis evite d'être sur le premier point en dehors de notre interval initial"""



    #ecartH.append(hMaxTri - hMinTri)

    tKernelTri = TriangularKernel(hMaxTri)

    for j in sample:
        hx += tKernelTri.value(x, j)
    yTriHMaxOnDomain.append(hx)



    """code ci-dessous à revoir"""

    """if ((hx - hx_moins_1) < 0.1):

        hMaxTri -= epsilonH
        compteurZero=0
        print("on a changé h max")"""

    """Retour aux Kernel avec les hTest défini initialement"""

    #tKernelTri = TriangularKernel(hTestTri)

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
#print(" moyenne ecart hMax - hMin :   " + str(stdev(ecartH)) + "    variance : " + str(variance(ecartH)))

#plt.plot(defDomain, yTriOnDomain, label="Brute Force Tri")
plt.plot(defDomain, yTriHMinOnDomain, label="TriHMin")
plt.plot(defDomain, yTriHMaxOnDomain, label="TriHMax")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.gca().set_position([0, 0, 0.8, 0.8])
plt.show()