import matplotlib.pylab as plt
import numpy as np
import random

from classes.Kernels.TriangularKernel import TriangularKernel
from classes.Extremizer.Extremizer import Extremizer
from classes.KernelContext import KernelContext

SEED = 1245632214124 #598473 # SUPER SEED ICI 157  #47 #53 # 23
random.seed(SEED)
experimentalSample = []
centerPoint = 5
for i in range(0,10):
    appending = False
    while appending == False:
        tirAlea = random.randint(0,20)
        if tirAlea != centerPoint:
            appending = True
            experimentalSample.append(tirAlea)

tKernel2 = TriangularKernel(0.1)

test = TriangularKernel.testUnitaires()

if test==0:

    dist2 = []
    extremizer = Extremizer(experimentalSample, centerPoint, tKernel2)
    maxStruct = extremizer.computeHMax()


    distances = []
    for ind,i in enumerate(experimentalSample):
        distance = abs(i-centerPoint)
        plt.axvline(x=distance*2, color='red')
        distances.append(distance*2)

    plt.axvline(x=maxStruct['potentialHValue'], color="green")

    #distances = sorted(distances)
    #for ind,p in enumerate(distances):
        #if ind != 9:
            #plt.axvline(x=(distances[ind+1]+p)/2, color='green')

    hDomain = np.linspace(0, 40, 200)

    hX = []
    tKernel = TriangularKernel(0.1)
    for ind,i in enumerate(hDomain):
        tKernel.bandwidth = i
        hX.append(0)
        for j in experimentalSample:
            hX[ind] += tKernel.value(centerPoint, j)

    plt.plot(hDomain, hX)
    plt.show()

    """sample = MultimodalGenerator([(100,-1,1),(400,5,2)]).generateNormalSamples()    #sample = echantillon
    hist, bins = np.histogram(sample, bins=50)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()"""
