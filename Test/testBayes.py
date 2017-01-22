import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import interp1d


from classes.Kernels.EpanechnikovKernel import EpanechnikovKernel
from classes.SampleGenerator.MultimodalGenerator import MultimodalGenerator
from classes.KernelContext import KernelContext

# On commence par générer des données
sample1 = MultimodalGenerator([(100,0,2),(100,10,2)]).generateNormalSamples()
sample2 = MultimodalGenerator([(100,5,2)]).generateNormalSamples()

# On sait que sur ce jeu de données, les 200 premières données sont de la classe A, et les 100 dernières sont de la classe 2
# Création de notre kernel
epaKernel = EpanechnikovKernel(2)

# On génère la fonction de densité de la classe A
AClassContext = KernelContext(sample1, epaKernel)
AClassDensity, AClassDensityDomain = AClassContext.computeTotalDensity()

# Pareil pour la classe B
BClassContext = KernelContext(sample2, epaKernel)
BClassDensity, BClassDensityDomain = BClassContext.computeTotalDensity()

# Pour voir ce que ça donne !
plt.plot(AClassDensityDomain, AClassDensity)
plt.plot(BClassDensityDomain, BClassDensity)

# Création des deux fonctions d'interpolation
fA = interp1d(AClassDensityDomain, AClassDensity, kind="cubic")
EstimatedA = fA(AClassDensityDomain)

fB = interp1d(BClassDensityDomain, BClassDensity, kind="cubic")
EstimatedB = fA(BClassDensityDomain)

AClassHMaxDensity = []
AClassHMinDensity = []
# On tente avec du HMax / HMin
for point in AClassDensityDomain:
    AClassHMaxDensity.append(AClassContext.computeHMaxFromInterval(point,2,0.2)['maxedValue'])
    AClassHMinDensity.append(AClassContext.computeHMinFromInterval(point,2,0.2)['minValue'])

# On génère les fonctions associées
fAMax = interp1d(AClassDensityDomain, AClassHMaxDensity, kind="cubic")
fAMin = interp1d(AClassDensityDomain, AClassHMinDensity, kind="cubic")

BClassHMaxDensity = []
BClassHMinDensity = []
# On tente avec du HMax / HMin
for point in BClassDensityDomain:
    BClassHMaxDensity.append(BClassContext.computeHMaxFromInterval(point,2,0.2)['maxedValue'])
    BClassHMinDensity.append(BClassContext.computeHMinFromInterval(point,2,0.2)['minValue'])

# On génère les fonctions associées
fBMax = interp1d(BClassDensityDomain, BClassHMaxDensity, kind="cubic")
fBMin = interp1d(BClassDensityDomain, BClassHMinDensity, kind="cubic")

# On regarde ce que ça donne
ATestSample = MultimodalGenerator([(10,0,2), (10,10,2)]).generateNormalSamples()
BTestSample = MultimodalGenerator([(20,5,2)]).generateNormalSamples()

supposedResult = []

for i in range(20):
    supposedResult.append("A")
for i in range(20):
    supposedResult.append("B")

results = []

for sample in ATestSample:
    try:
        probaA = fAMin(sample)
    except:
        probaA = 0

    try:
        probaB = fBMax(sample)
    except:
        probaB = 0

    try:
        finalProba = probaA * 2/3 / (probaB * 1/3)
    except:
        finalProba = 0

    if finalProba > 1:
        results.append("A")
    else:
        results.append("IND")

for sample in BTestSample:
    try:
        probaB = fBMin(sample)
    except:
        probaB = 0

    try:
        probaA = fAMax(sample)
    except:
        probaA = 0

    try:
        finalProba = probaB * 1/3 / (probaA * 2/3)
    except:
        finalProba = 0

    if finalProba > 1:
        results.append("B")
    else:
        results.append("IND")

print(results)

error = 0
for i in range(40):
    if supposedResult[i] != results[i]:
        error += 1

error = error/40
print(error)

plt.show()