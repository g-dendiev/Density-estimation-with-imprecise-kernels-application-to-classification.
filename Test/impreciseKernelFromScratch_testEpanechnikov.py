# NAIVE BAYE SFROM SCRATCH AVEC : https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/

import matplotlib.pylab as plt
import numpy as np

from statistics import stdev


from classes.Kernels.EpanechnikovKernel import EpanechnikovKernel
from classes.KernelContext import KernelContext

# 1 ) On importe les données

import csv
def loadCsv(filename):
	lines = csv.reader(open(filename,"rt"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in  dataset[i]]
	return dataset

# Test import : IT WORKS

#filename = 'pima-indians-diabetes.data.csv'
#dataset = loadCsv(filename=filename)
#print('load data au format',filename, 'avec', len(dataset) ,'lignes ')

# Séparation des données

import random
def splitDataset(dataset,splitRatio):
	trainSize = int(len(dataset)*splitRatio)
	trainSet = []
	copy = list(dataset)
	while (len(trainSet)<trainSize):
		index=random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet,copy]

# Test split : IT WORKS

#dataset = [[1],[2],[3],[4],[5]]
#splitRatio = (2/3)
#train, test = splitDataset(dataset=dataset, splitRatio=splitRatio)
#print('on a split avec en test :',test,'et en train : ',train)


# 2 ) Sommaire des données

# Separation des données

# On ajoute une variable qui définit la colonne qui contient la réponse = la classe dans le dataset
def separateByClass(dataset,columnWithClassResponse):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		# On crée la ligne de la classe si elle existe pas
		if ( vector[columnWithClassResponse] not in separated):
			separated[vector[columnWithClassResponse]]=[]
		separated[vector[columnWithClassResponse]].append(vector)
	return separated

# Test separation des donnees : IT WORKS

#dataset = [[1,20,1],[2,34,4],[1,34,4],[2,20,1]]
#separated0 = separateByClass(dataset=dataset, columnWithClassResponse=0)
#separated1 = separateByClass(dataset=dataset, columnWithClassResponse=1)
#separated_1 = separateByClass(dataset=dataset, columnWithClassResponse=-1)
#print('separation 1 sur col 0', separated0)
#print('separation 2 sur col 1', separated1)
#print('separation 3 sur col -1', separated_1)



# ADAPTATION D UNE PARTIE DU CODE POUR L UTILISATION DE NOTRE KERNEL IMPRECIS :
# 3 ) Make predictions :

# ON UTILISE LES FONCTIONS DU KERNEL PLUTOT QUE CELLE LA POUR NOS ESTIMATION

def InitHOptKernelImprecise(dataset):
	from statistics import stdev
	sigma=stdev(dataset)
	hOpt = 1.06 * sigma * (len(dataset)) ** (-1 / 5)
	return hOpt


#Cette fonction est la MAJ avec les fonctions de notre kernel !
#import math
# Ici on devra faire 2 parties :
# une pour proba haute et une autre pour proba basse

import math
def calculsDi(x,Xi):
	sumDi = 0
	#print('Xi = ',Xi)
	#print('x= ',x)
	for i in range(len(Xi)):
		sumDi += abs(x-Xi[i])
	meanDi = sumDi/len(Xi)
	return sumDi, meanDi

#CALCUL PROBA CONDITIONNELLE
def findProbabilityImpreciseKernel(x,generalHightProbabilities,tKernelDomain, generalLowProbabilities):
	iMin = 0
	#print('VALEUR DE X : ', x)
	#print(' PROBAS MAXIMALES :',generalHightProbabilities)
	#print(' PROBAS MINIMALES :', generalLowProbabilities)
	#print(' TKERNEL DOMAIN :', tKernelDomain)
	for i in range(len(tKernelDomain)):
			tKernelDomain[i] = abs(tKernelDomain[i] - x)
			if (tKernelDomain[iMin]>tKernelDomain[i]):
				iMin = i
	return generalLowProbabilities[iMin],generalHightProbabilities[iMin]




def calculateProbabilityImpreciseKernel(dataset, h,epsilon,N):
	# TRIER LES DONNEES DU DATASET ET ENSUITE MAJ LA SOMME A CHAQUE ITERATION :)
	#sortedDataset = sorted(dataset)
	n = len(dataset)
	#print('n=', n)
	stepLinspace = 0.1

	lowProbabilities = []
	hightProbabilities = []
	#print('DATASET AVANT PASSAGE DANS KERNEL :', dataset)
	tKernelTri = KernelContext(dataset, EpanechnikovKernel(h), stepLinspace)

	for pt in tKernelTri.domain:
		# Def des structures qui vont récolter les données (dans la boucle pour une remise à 0 à chaque cycle
		#lenDomain.append(len(tKernelTri.domain))

		structHMin = {'potentialHValue': -1, 'minValue': -1}

		structHMax = {'potentialHValue': -1, 'maxedValue': -1}

		# Calculs de f(hMax), et f(hMin)
		structHMax = tKernelTri.computeHMaxFromInterval(pt, h, epsilon)
		structHMin = tKernelTri.computeHMinFromInterval(pt, h, epsilon)

		hightProbabilities.append(structHMax['maxedValue'])
		lowProbabilities.append(structHMin['minValue'])

	#print('Hight probability = ', hightProbability)
	return lowProbabilities, hightProbabilities, tKernelTri.domain

# Test calcul proba : IT WORKS !
#x = 71.5
#mean = 73
#stdev = 6.2
#frequence_y=0.5
#lowProbability = calculateLowProbability(x=x,mean=mean,stdev=stdev)
#hightProbability = calculateHightProbability(x=x,mean=mean,stdev=stdev)
#print('Proba de x = 71.5 d appartenir a la classe de moyenne 71.5 et stdev 6.2 selon Naive Bayes:',probability)


# Separation des classes en enlevant la colonne de la réponse:
# On ajoute une variable qui définit la colonne qui contient la réponse = la classe dans le dataset
# columnWithClassResponse doit être la premiere colonne (0) ou la dernière (-1)
def separateByClassWithoutResponse(dataset,columnWithClassResponse):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		# On crée la ligne de la classe si elle existe pas
		if ( vector[columnWithClassResponse] not in separated):
			separated[vector[columnWithClassResponse]] = []
		if (columnWithClassResponse == -1):
			separated[vector[columnWithClassResponse]].append(vector[0:columnWithClassResponse]) # On supprime la var avec la classe
		else:
			separated[vector[columnWithClassResponse]].append(vector[1:])  # On supprime la var avec la classe
	return separated


# Calcul des probas de classes maintenant !

'''
Now that we can calculate the probability of an attribute belonging to a class,
we can combine the probabilities of all of the attribute values for a data instance
and come up with a probability of the entire data instance belonging to the class.

We combine probabilities together by multiplying them. In the calculateClassProbabilities() below,
the probability of a given data instance is calculated by multiplying together
the attribute probabilities for each class. the result is a map of class values to probabilities.
'''

# FAIRE UNE INITIALISATION DES PARAMETRES POUR KERNEL IMPRECIS AVANT CETTE FONCTION
# PASSER EN PARAMETRE CES DONNES POUR LA FONCTION SUIVANTE AFIN DE POUVOIR LANCER ComputeHMaxFromInterval

def calculateClassProbabilitiesImpreciseKernel(dataset,columnWithClassResponse,inputVector,margeEpsilon): #,columnWithClassResponse): Pas important ici je pense
	N = len(dataset)
	separated = separateByClassWithoutResponse(dataset,columnWithClassResponse)
	dataset2 = []
	hOpt=[]
	for i in range(len(dataset)):
		if(columnWithClassResponse == -1):
			dataset2.append(dataset[i][0:columnWithClassResponse])
		else:
			dataset2.append(dataset[i][1:])
	dataset2Separated = [attribute1 for attribute1 in zip(*dataset2)]
	for i in range(len(dataset2Separated)):
		hOpt.append(InitHOptKernelImprecise(dataset2Separated[i]))
	#print(len(separated.items()))
	lowProbabilities = {}
	hightProbabilities = {}
	for classValue,classDataset in separated.items():
		#print('classDataset :',classDataset)
		#print('len classDataset',len(classDataset))
		# Separation des colonnes pour qu'une colonne corresponde à une seule variable d'entrée
		#print('zip : ',zip(*classDataset))
		classDatasetWithColSeparated = [attribute for attribute in zip(*classDataset)]
		#print('class : ',classDatasetWithColSeparated)
		#print(len(classDatasetWithColSeparated))
		lowProbabilities[classValue] = 1 #Initialisation pour la multiplication ensuite des probas
		hightProbabilities[classValue] = 1  # Initialisation pour la multiplication ensuite des probas
		frequence_y = len(classDataset)/N
		for i in range(len(classDatasetWithColSeparated)):
			#print(i)
			x = inputVector[i]
			#print(x)
			#Calcul toute la densité sur la classe et ensuite on assigne les prbas selon les points en entrée et en sortie
			generalLowProbabilities, generalHightProbabilities, generalDomain = calculateProbabilityImpreciseKernel(dataset=classDatasetWithColSeparated[i], h=hOpt[i],epsilon=margeEpsilon*hOpt[i],N=N)
			lowProbability,hightProbability = findProbabilityImpreciseKernel(x,generalHightProbabilities,generalDomain, generalLowProbabilities)
			lowProbabilities[classValue] *=  lowProbability
			lowProbabilities[classValue] *= frequence_y  # multiplication par l'estimation de p(y)
			hightProbabilities[classValue] *= hightProbability
			hightProbabilities[classValue] *= frequence_y # multiplication par l'estimation de p(y)
			#print(' frequence d apparition de la Classe value = ',classValue,'est de :',frequence_y,', avec une Probabilité haute = ',hightProbabilities[classValue],', et une probabilité basse :', lowProbabilities[classValue])
	return lowProbabilities, hightProbabilities

#testLow=calculateClassLowProbabilitiesImpreciseKernel(dataset=[[1,3,2,5],[2,6,3,5],[1,4,4,5]],columnWithClassResponse=0,inputVector=[4,7,7],hOpt=10,margeEpsilon=0.5)
#print('testLow = ',testLow)

'''
def calculateClassHightProbabilitiesImpreciseKernel(dataset,columnWithClassResponse,inputVector,margeEpsilon): #,columnWithClassResponse): Pas important ici je pense
	N = len(dataset)
	dataset2 = []
	hOpt = []
	separated = separateByClassWithoutResponse(dataset, columnWithClassResponse)
	for i in range(len(dataset)):
		if (columnWithClassResponse == -1):
			dataset2.append(dataset[i][0:columnWithClassResponse])
		else:
			dataset2.append(dataset[i][1:])
	dataset2Separated = [attribute1 for attribute1 in zip(*dataset2)]
	for i in range(len(dataset2Separated)):
		hOpt.append(InitHOptKernelImprecise(dataset2Separated[i]))
		# print(len(separated.items()))
	hightProbabilities = {}
	for classValue, classDataset in separated.items():
		#print('classDataset :', classDataset)
		#print('len classDataset', len(classDataset))
		# Separation des colonnes pour qu'une colonne corresponde à une seule variable d'entrée
		# print('zip : ',zip(*classDataset))
		classDatasetWithColSeparated = [attribute for attribute in zip(*classDataset)]
		# print('class : ',classDatasetWithColSeparated)
		# print(len(classDatasetWithColSeparated))
		hightProbabilities[classValue] = 1  # Initialisation pour la multiplication ensuite des probas
		frequence_y = len(classDataset) / N
		for i in range(len(classDatasetWithColSeparated)):
			#print(i)
			x = inputVector[i]
			#print(x)
			## CALCUL MEANDI ET SUMDI ET ENSUITE ON ENVOIE !
			# mean, stdev, frequence_y = classSummaries[i]
			hightProbabilities[classValue] *= calculateHightProbabilityImpreciseKernel(x=x,dataset=classDatasetWithColSeparated[i], h=hOpt[i], epsilon=margeEpsilon*hOpt[i], N=N)
			hightProbabilities[classValue] *= frequence_y  # multiplication par l'estimation de p(y)
	return hightProbabilities
'''
#testHight=calculateClassHightProbabilitiesImpreciseKernel(dataset=[[1,3,2,5],[2,6,3,5],[1,4,4,5]],columnWithClassResponse=0,inputVector=[4,7,7],hOpt=10,margeEpsilon=0.3)
#print('testhight = ',testHight)

#Test proba par classe : IT WORKS
#summaries = {0:[(1, 0.5,.8)], 1:[(20, 5.0,.2)]}
#inputVector = [1.1] # pas besoin ici de mettre une fausse valeur en y ou même de faire inputVector = [1.1, ]
#probabilities = calculateClassProbabilities(summaries, inputVector)
#print('Probabilities for each class: ',probabilities)

# 4 ) Prédictions !

# Retourner 1 prediction avec 1 ou plusieurs classes :

def predictImpreciseKernel(dataset,columnWithClassResponse,inputVector,margeEpsilon):
	#calculateClassProbabilitiesImpreciseKernel(dataset,columnWithClassResponse,inputVector,margeEpsilon):
	lowProbabilities, hightProbabilities = calculateClassProbabilitiesImpreciseKernel(dataset,columnWithClassResponse,inputVector,margeEpsilon)
	bestLabel = []
	#print('prediction de low probabilities : ',lowProbabilities)
	for classValueLow,lowProba in lowProbabilities.items():
		#print('class Value = ',classValueLow)
		#print('lowProba =',lowProba)
		tjrsSup = 1
		for classValueHight,hightProba in hightProbabilities.items():
			if classValueHight != classValueLow :
				#print('hightProba = ',hightProba)
				if lowProba > hightProba and tjrsSup == 1:
					tjrsSup = 1
				else:
					tjrsSup = 0
		if(tjrsSup == 1):
			bestLabel.append(classValueLow)
	if(bestLabel == []):
		#On teste si notre proba haute est inf à toutes les probas basses
		#Si c'est le cas on ne met pas la classe dans les potentielles classes de retour
		#Sinon on ajoute la classe aux classes de retour
		#print('passage par le deuxième cycle de comparaison avec pour hight :',hightProbabilities,' et pour low : ',lowProbabilities)
		for classValueHight2, hightProba2 in hightProbabilities.items():
			#print('2e partie : class Value = ', classValueHight2)
			#print('2e partie : hightProba =', hightProba2)
			tjrsInf = 1
			for classValueLow2, lowProba2 in lowProbabilities.items():
				if classValueHight2 != classValueLow2:
					#print('2e partie : lowProba = ', lowProba2)
					if lowProba2 > hightProba2 and tjrsInf == 1:
						tjrsInf = 1
					else:
						tjrsInf = 0
			if (tjrsInf != 1):
				bestLabel.append(classValueHight2)

	return bestLabel

# Test prediction : IT WORKS

#testPredict=predictImpreciseKernel(dataset=[[1,3,2,5],[2,6,3,5],[1,3.5,2.2,5.1],[3,3,2,5],[3,3.5,2.2,5.1]],columnWithClassResponse=0,inputVector=[3.5,2.8,6],hOpt=5,margeEpsilon=0.1)
#print('testpredict = ',testPredict)

# Predictions sur un jeu de test complet :
def getPredictionsImpreciseKernel(dataset,columnWithClassResponse,testSet,margeEpsilon):
	predictions = []
	#print('test set passé en argument =',testSet)
	for i in range(len(testSet)):
		#print('test set de ',i,' = ',testSet[i])
		result = predictImpreciseKernel(dataset,columnWithClassResponse,testSet[i], margeEpsilon)
		#print('resultat pour la ligne ',i,' : ',result)
		predictions.append(result)
	return predictions

# Test : IT WORKS
#testSet=[[3.5,2.8,6],[3,3,5],[10.1,12.1,14.1]]
#testPredictions=getPredictionsImpreciseKernel(dataset=[[1,3,2,5],[2,10,12,14],[1,3.5,2.2,5.1],[3,3,2,5],[3,3.5,2.2,5.1]],columnWithClassResponse=0,testSet=testSet,hOpt=5,margeEpsilon=0.1)
#print('testpredictions = ',testPredictions)

# 5 ) Moyenne des erreurs :

def getAccuracyImpreciseKernel(testSet, predictions, columnWithClassResponse):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][columnWithClassResponse] in predictions[x]:
			correct += 1/len(predictions[x])
	return (correct/float(len(testSet))) * 100.0

# Test :
#testSet = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
#predictions = ['a', 'a', ['a','b']]
#accuracy = getAccuracyImpreciseKernel(testSet, predictions,3)
#print('Accuracy: ',accuracy)





# CODE POUR LANCER LES FONCTIONS ET PREDIRE :

def main():
	file = 'iris.data.csv'
	splitRatio = 0.20
	dataset = loadCsv(file)
	trainingSet, testSet = splitDataset(dataset, splitRatio)
	# prepare model
	print('Split ',len(dataset),' rows into train=',len(trainingSet),' and test=',len(testSet),' rows')
	testSet2 = []
	valeurAttendue = []
	for i in range(len(testSet)):
		testSet2.append(testSet[i][0:4])
		valeurAttendue.append([testSet[i][4]])
	# test model
	predictionsPK = getPredictionsImpreciseKernel(dataset, columnWithClassResponse=-1,testSet=testSet2,margeEpsilon=0)
	#La colonne avec la réponse de classe doit être 0 ou -1 (1ere ou dernière colonne du dataset passé en parametre
	predictionsIK = getPredictionsImpreciseKernel(dataset, columnWithClassResponse=-1,testSet=testSet2,margeEpsilon=0.6)
	print('predictions PK =', predictionsPK)
	print('valeur atendue :',valeurAttendue)
	print('predictions IK =',predictionsIK) #le zip permet d'enlever les sous-tableaux quand on a 1 seule valeur
	accuracyPK = getAccuracyImpreciseKernel(testSet, predictionsPK,(4))
	accuracyIK = getAccuracyImpreciseKernel(testSet, predictionsIK,(4))
	print('Accuracy Precise Kernel : ',accuracyPK)
	print('Accuracy Imprecise Kernel : ',accuracyIK)
#main()

def launchXTimes(times):
	for i in range(times):
		print('Resultats de l\'iteration : ', i+1)
		main()

launchXTimes(1)


# Tableaau des identifiants des réponses :
#0 = setosa
#1 = versicolor
#2 = virginica