# TEST CHANGEMENT NB POINTS POUR REGRESSION

import matplotlib.pylab as plt
import numpy as np
import random as rd

from statistics import stdev


from classes.Kernels.TriangularKernel import TriangularKernel
from classes.KernelContext import KernelContext
from classes.SampleGenerator.MultimodalGenerator import MultimodalGenerator

# Def le nombre de points pour le linspace et pour la multimodale

nbPointsFirstGauss = 900
nbPointsSecondGauss = 600

# Def du step pour la génération du linspace dans KernelContext
stepLinspace = 0.1


# Génération multimodale

sample = MultimodalGenerator([(nbPointsFirstGauss,-1,1),(nbPointsSecondGauss,5,2)]).generateNormalSamples()

#print(sample)

# Def du hOpt

sigma=stdev(sample)

hOpt = 1.06*sigma*(nbPointsFirstGauss+nbPointsSecondGauss)**(-1/5)
#print("hopt",hOpt)

# Def epsilon

epsilon = 0.2*hOpt   #mettre ce paramettre en fonction du hOpt trouvé.


#def Kernel

tKernelTri = KernelContext(sample,TriangularKernel(hOpt),stepLinspace)
#On melange aleatoirement les données du dataset.
#On garde avant les données dans notre var contenant toutes les vars du dataset.
sample1500 = tKernelTri.dataset

rd.shuffle(tKernelTri.dataset)

#On construit nos jeux de donnees avec n points tirer aleatoirement suite au tri precedent.
#Ensuite on range les pints dans l'ordre croissant pour la regression.
sample25 = tKernelTri.dataset[0:25]
sample50 = tKernelTri.dataset[0:50]
sample100 = tKernelTri.dataset[0:100]
sample200 = tKernelTri.dataset[0:200]
sample500 = tKernelTri.dataset[0:500]
# Matrice contenant l'ensemble des sous ensembles de points
samples = [sample25,sample50,sample100,sample200,sample500,sample1500]


for i in range(6):
    # Initialisation du dataset + rangement par ordre croissant des données
    tKernelTri.dataset=samples[i]
    tKernelTri.dataset.sort()

    # Def des tableau qui vont stocker les valeurs des f(hMax), f(hOpt) et f(hMin)
    # for domain in (sample25):#, sample50, sample100, sample200, sample500, sample1500):
    # print(domain)
    yTriHOptOnDomain = []
    yTriHMaxOnDomain = []
    yTriHMinOnDomain = []

    yInitialBimodal = []

    lenDomain = []

    nbPointsTot = len(tKernelTri.dataset)

    for pt in tKernelTri.domain:
        # Def des structures qui vont récolter les données (dans la boucle pour une remise à 0 à chaque cycle
        lenDomain.append(len(tKernelTri.domain))

        structHOpt = {
            'potentialHValue': -1,
            'minValue': -1
        }

        structHMin = {
            'potentialHValue': -1,
            'minValue': -1
        }

        structHMax = {
            'potentialHValue': -1,
            'minValue': -1
        }

        # Calculs de f(hOpt),f(hMax), et f(hMin)

        structHOpt = tKernelTri.computeHMaxFromInterval(pt,hOpt,0)
        structHMax = tKernelTri.computeHMaxFromInterval(pt,hOpt,epsilon)
        structHMin = tKernelTri.computeHMinFromInterval(pt,hOpt,epsilon)

        yTriHOptOnDomain.append(structHOpt['maxedValue'])
        yTriHMaxOnDomain.append(structHMax['maxedValue'])
        yTriHMinOnDomain.append(structHMin['minValue'])

        #print("yTriHOptOnDomain", yTriHOptOnDomain)

    plt.title("Curves obtained with %d points using the triangular kernel. \n hOpt = %.3g, hMax and hMin in [hOpt - %.3g, hOpt + %.3g]\n" % (nbPointsTot, hOpt, epsilon, epsilon))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(tKernelTri.domain, yTriHOptOnDomain, label="RegHOpt")
    plt.plot(tKernelTri.domain, yTriHMaxOnDomain, label="RegHMax")
    plt.plot(tKernelTri.domain, yTriHMinOnDomain, label="RegHMin")
    plt.xlim(-5, 10)
    plt.legend(loc="upper right")  # loc=2, borderaxespad=0., bbox_to_anchor=(.5, 1)
    # plt.gca().set_position([0, 0, 0.8, 0.8])
    plt.show()
print("fin du programme")

    # fonction initiale

#yInitialBimodal.append(tKernelTri.dataset)

"""
print("hOpt tableau")
print(yTriHOptOnDomain)

print("taille hOpt tableau")
print(len(yTriHOptOnDomain))

print("hMax tableau")
print(yTriHMaxOnDomain)

print("taille hOpt tableau")
print(len(yTriHMaxOnDomain))

print("hMin tableau")
print(yTriHMinOnDomain)

print("taille hOpt tableau")
print(len(yTriHMinOnDomain))

x=np.array(tKernelTri.domain)
y1=np.array(yTriHOptOnDomain)
y2=np.array(yTriHMaxOnDomain)
y3=np.array(yTriHMinOnDomain)

print("taille domaine",x)
print("taille tKernel.Domain",len(x))

"""
#plt.plot(tKernelTri.domain,yInitialBimodal, label="Initial bimodal")
#plt.title("Curves obtained with %d points using the triangular kernel. \n hOpt = %.3g, hMax and hMin in [hOpt - %.3g, hOpt + %.3g]\n" %(nbPointsTot,hOpt, epsilon, epsilon))
#plt.xlabel("x")
#plt.ylabel("y")
#plt.plot(tKernelTri.domain, yTriHOptOnDomain, label="RegHOpt")
#plt.plot(tKernelTri.domain, yTriHMaxOnDomain, label="RegHMax")
#plt.plot(tKernelTri.domain, yTriHMinOnDomain, label="RegHMin")
#plt.xlim(-5,10)
#plt.legend(loc="upper right") #loc=2, borderaxespad=0., bbox_to_anchor=(.5, 1)
#plt.gca().set_position([0, 0, 0.8, 0.8])
#plt.show()


