import matplotlib.pylab as plt
import numpy as np
import random

from classes.Kernels.TriangularKernel import TriangularKernel

SEED = 53 # 23
random.seed(SEED)
experimentalSample = []
for i in range(0,10):
    experimentalSample.append(random.randint(0,20))

tentativeDesesperee = sum(experimentalSample)/len(experimentalSample)
experimentalSample = sorted(experimentalSample)
print(experimentalSample)
centerPoint = 5
print(abs(tentativeDesesperee-centerPoint)*4)
hMin = 1000000
hMax = 0
distances = []
for ind,i in enumerate(experimentalSample):
    distance = abs(i-centerPoint)
    if  hMin > distance:
        hMin = distance
    if hMax < distance*2:
        hMax = distance*2
    plt.axvline(x=distance*2, color='red')
    distances.append(distance*2)

distances = sorted(distances)
#for ind,p in enumerate(distances):
    #if ind != 9:
        #plt.axvline(x=(distances[ind+1]+p)/2, color='green')

hDomain = np.linspace(hMin*2, hMax*2+4, 100)

hX = []
tKernel = TriangularKernel(0.1)
for ind,i in enumerate(hDomain):
    tKernel.bandwidth = i
    hX.append(0)
    for j in experimentalSample:
        hX[ind] += tKernel.value(centerPoint, j)

print(max(hX))

plt.plot(hDomain, hX)
plt.show()

"""sample = MultimodalGenerator([(100,-1,1),(400,5,2)]).generateNormalSamples()    #sample = echantillon
hist, bins = np.histogram(sample, bins=50)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.show()"""