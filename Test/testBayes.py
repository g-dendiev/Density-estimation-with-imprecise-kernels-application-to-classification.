import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import interp1d
from sklearn.naive_bayes import GaussianNB


from classes.Kernels.EpanechnikovKernel import EpanechnikovKernel
from classes.SampleGenerator.MultimodalGenerator import MultimodalGenerator
from classes.KernelContext import KernelContext

totalErrorKernel = 0
totalErrorNB = 0
totalErrorKernelPrecis = 0

nbTotalIterations = 100

for e in range(nbTotalIterations):
    # On commence par générer des données
    sample1 = MultimodalGenerator([(100,2,2),(100,8,2)]).generateNormalSamples()
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

    # On génére notre modèle NB
    gnb = GaussianNB()
    sample = []
    sample.extend(sample1)
    sample.extend(sample2)
    labels = []
    for i in range(200):
        labels.append("A")
    for i in range(100):
        labels.append("B")

    GNBModel = gnb.fit(np.array(sample).reshape(300,1), np.array(labels).reshape(300,1))

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

    # Nombre pair !
    nbTestDataClassA = 20
    nbTestDataClassB = 20

    nbTotalTestData = nbTestDataClassA + nbTestDataClassB

    # On regarde ce que ça donne
    testSample = MultimodalGenerator([(nbTestDataClassA/2,0,2), (nbTestDataClassA/2,10,2), (nbTestDataClassB,5,2)]).generateNormalSamples()

    supposedResult = []

    for i in range(nbTestDataClassA):
        supposedResult.append("A")
    for i in range(nbTestDataClassB):
        supposedResult.append("B")

    results = []
    resultsPrecis = []

    for sample in testSample:
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
            resultsPrecis.append("A")
        else:
            resultsPrecis.append("B")
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

    pred = gnb.predict(np.array(testSample).reshape(40,1))

    errorKernel = 0
    errorNB = 0
    errorKernelPrecis = 0
    for i in range(nbTotalTestData):
        if results[i] == 'IND':
            errorKernel += 0.5
        elif supposedResult[i] != results[i]:
            errorKernel += 1

        if supposedResult[i] != resultsPrecis[i]:
            errorKernelPrecis += 1

        if pred[i] != supposedResult[i]:
            errorNB += 1

    errorKernel = errorKernel/nbTotalTestData
    totalErrorKernel += errorKernel

    errorKernelPrecis = errorKernelPrecis / nbTotalTestData
    totalErrorKernelPrecis += errorKernelPrecis

    errorNB = errorNB/nbTotalTestData
    totalErrorNB += errorNB


totalErrorKernel = totalErrorKernel/nbTotalIterations
totalErrorNB = totalErrorNB/nbTotalIterations
totalErrorKernelPrecis = totalErrorKernelPrecis/nbTotalIterations

print("Erreur du kernel Imprecis : "+str(totalErrorKernel))
print("Erreur du kernel Précis : "+str(totalErrorKernelPrecis))
print("Erreur du Naive Bayes : "+str(totalErrorNB))