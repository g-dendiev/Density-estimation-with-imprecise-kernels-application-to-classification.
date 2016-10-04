import matplotlib.pylab as plt
import numpy as np

from classes.Kernels.TriangularKernel import TriangularKernel
from classes.SampleGenerator.MultimodalGenerator import MultimodalGenerator

sample = MultimodalGenerator([(100,-1,1),(400,5,2)]).generateNormalSamples()
tKernel = TriangularKernel()
defDomain = np.linspace(-4,12,200)

yOnDomain = []
for x in defDomain:
    fx = 0
    for j in sample:
        fx += tKernel.value(x,j)
    yOnDomain.append(fx)

plt.plot(defDomain, yOnDomain)
plt.show()