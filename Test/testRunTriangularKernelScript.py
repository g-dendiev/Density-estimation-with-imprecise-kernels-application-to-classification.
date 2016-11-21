import matplotlib.pylab as plt
import numpy as np

from classes.Kernels.TriangularKernel import TriangularKernel
from classes.SampleGenerator.MultimodalGenerator import MultimodalGenerator

""" Cf readme pour les valeurs de hMin et hMax"""
hMinTri = 0.01
hMaxTri = 100

"""ici c'est juste pour tester : la valeur de epsilon, borne est arbitraire, il faudra les optimiser"""
epsilonHMin = 0.05
epsilonHMax = 2
ecartMax = 10
diffMin = 0.01

"""Déclaration et initialisation d'un compteur"""
compteur = 0

"""Génération de la multimodale """
sample = MultimodalGenerator([(100,-1,1),(400,5,2)]).generateNormalSamples()

"""Def du domaine d'étude, on a ici 200 points entre -4 et 12"""
defDomain = np.linspace(-4,12,200)

"""Définition des tableaux de valeurs pour nos tests"""
yTriHMinOnDomain = []
yTriHMaxOnDomain = []

"""Nombre de 0 successifs autorisés"""
nbZeroSuccHMin = 1
nbZeroSuccAutoris = 3


"""Var pour avoir l'écart entre 2 points successifs"""
gx_moins_1 = 0
hx_moins_1 = 0


for x in defDomain:


    if (compteur >= 1):
        """on sauv la dernière valeur de chaque variable : gx et hx"""
        gx_moins_1 = gx
        hx_moins_1 = hx

    """remise à zéro de nos variables pour remplir les tableaux de valeurs de f(hMin) et f(hMax)"""
    fx = 0
    gx = 0
    hx = 0

    """Incrémantation du compteur pour savoir si on peut sauver ou non l'ancienne valeur"""
    compteur += 1




    """Ici on passe aux Kernel avec un h minimal pour le point x dans le domaine de definition"""

    tKernelTri = TriangularKernel(hMinTri)


    for j in sample:
        gx += tKernelTri.value(x,j)
    yTriHMinOnDomain.append(gx)

    """La partie qui suit avec le changement de hMin va etre fait sous forme d'une fonction plus tard"""

    if (abs(gx - gx_moins_1) >= (ecartMax)):
        """On modifie hMin en faisant hMin + epsilon"""
        print("On a changé hMin on lui a ajouté epsilon suite à de trop grands écart entre 2 itérations")
        hMinTri += epsilonHMin
    else:
        if (gx_moins_1 == gx == 0):
            """Deux zéros successifs de suite"""
            nbZeroSuccHMin += 1

            if (nbZeroSuccHMin >= nbZeroSuccAutoris):
                print("On a changé hMin on lui a ajouté epsilon suite à des valeurs nulles successives")
                hMinTri += epsilonHMin
                nbZeroSuccHMin = 1
        else:
            nbZeroSuccHMin = 1






    """Ici on passe aux Kernel avec un h maximal pour le point x dans le domaine de definition"""




    tKernelTri = TriangularKernel(hMaxTri)

    for j in sample:
        hx += tKernelTri.value(x, j)
    yTriHMaxOnDomain.append(hx)

    if (abs(hx - hx_moins_1) <= (diffMin)):
        """On modifie hMin en faisant hMin + epsilon"""
        print("On a changé hMax on lui a ajouté epsilon suite à de trop grands écart entre 2 itérations")
        hMaxTri -= epsilonHMax





plt.figure(figsize=(10,8))

"""Histogramme pour voir la tête de la répartition"""
hist, bins = np.histogram(sample, bins=15)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.subplot(223)
barlist = plt.bar(center, hist, align='center', width=width)
for bar in barlist:
    bar.set_color('y')

print(round(hMinTri,3), hMaxTri)
plt.plot(defDomain, yTriHMinOnDomain, label="TriHMin")
plt.plot(defDomain, yTriHMaxOnDomain, label="TriHMax")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.gca().set_position([0, 0, 0.8, 0.8])
plt.show()