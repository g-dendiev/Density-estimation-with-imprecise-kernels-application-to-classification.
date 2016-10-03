import numpy as np
import matplotlib.pylab as plt

mu1, sigma1, n1 = 0, 1, 275
mu2, sigma2, n2 = 4, 1, 500

echantillon = np.random.normal(mu1, sigma1, n1)
echantillon = np.append(echantillon, np.random.normal(mu2, sigma2, n2))

hist, bins = np.histogram(echantillon, bins=50)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.show()