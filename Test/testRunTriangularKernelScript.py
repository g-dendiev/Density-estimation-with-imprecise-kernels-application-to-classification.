import matplotlib.pylab as plt
import numpy as np

from statistics import stdev


from classes.Kernels.TriangularKernel import TriangularKernel
from classes.KernelContext import KernelContext
from classes.SampleGenerator.MultimodalGenerator import MultimodalGenerator

# Def le nombre de points pour le linspace et pour la multimodale

nbPointsFirstGauss = 40
nbPointsSecondGauss = 200

# Def du step pour la génération du linspace dans KernelContext

stepLinspace = 0.1

# Def epsilon

epsilon = 1     #mettre ce paramettre en fonction du hOpt trouvé.

# Génération multimodale

sample = MultimodalGenerator([(nbPointsFirstGauss,-1,1),(nbPointsSecondGauss,5,2)]).generateNormalSamples()

# Def du hOpt

sigma=stdev(sample)

hOpt = 1.06*sigma*(nbPointsFirstGauss+nbPointsSecondGauss)**(-1/5)
#print("hopt",hOpt)

#def Kernel

tKernelTri = KernelContext(sample,TriangularKernel(hOpt),stepLinspace)

# Def des tableau qui vont stocker les valeurs des f(hMax), f(hOpt) et f(hMin)

yTriHOptOnDomain = []
yTriHMaxOnDomain = []
yTriHMinOnDomain = []

lenDomain=[]

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

"""
print("hOpt tableau ma gueule")
print(yTriHOptOnDomain)

print("taille hOpt tableau ma gueule")
print(len(yTriHOptOnDomain))

print("hMax tableau ma gueule")
print(yTriHMaxOnDomain)

print("taille hOpt tableau ma gueule")
print(len(yTriHMaxOnDomain))

print("hMin tableau ma gueule")
print(yTriHMinOnDomain)

print("taille hOpt tableau ma gueule")
print(len(yTriHMinOnDomain))

x=np.array(tKernelTri.domain)
y1=np.array(yTriHOptOnDomain)
y2=np.array(yTriHMaxOnDomain)
y3=np.array(yTriHMinOnDomain)

print("taille domaine",x)
print("taille tKernel.Domain",len(x))

"""

plt.plot(tKernelTri.domain, yTriHOptOnDomain, label="TriHOpt")
plt.plot(tKernelTri.domain, yTriHMaxOnDomain, label="TriHMax")
plt.plot(tKernelTri.domain, yTriHMinOnDomain, label="TriHMin")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.gca().set_position([0, 0, 0.8, 0.8])
plt.show()


