import matplotlib.pylab as plt
import numpy as np

from classes.SampleGenerator.MultimodalGenerator import MultimodalGenerator

sample = MultimodalGenerator([(100,-1,1),(400,5,2)]).generateNormalSamples()    #sample = echantillon
hist, bins = np.histogram(sample, bins=50)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.show()