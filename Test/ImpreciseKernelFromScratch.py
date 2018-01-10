# NAIVE BAYE SFROM SCRATCH AVEC : https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/


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


# Calcul de données de stats :

import math
def mean(numbers):
	return sum(numbers)/len(numbers)

def stdev2(numbers):
	avg = mean(numbers=numbers)
	variance = sum([pow(x-avg,2)for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

# Test : IT WORKS
#numbers = [1,2,3,4,5]
#print('moyenne',mean(numbers=numbers))
#print('variance', stdev(numbers=numbers))

#Sommaire des données : mode général :
#The zip function groups the values for each attribute across our data instances
# #into their own lists so that we can compute the mean and standard deviation values for the attribute.
# On ajoute la frequence de la classe y comme approximation de p(y)
def summarize(dataset,columnWithClassResponse,frequence_y):
	summary = [(mean(attribute), stdev2(attribute),frequence_y) for attribute in zip(*dataset)]
	del summary[columnWithClassResponse]
	return summary

# Test summarize : IT WORKS
#dataset = [[1,20,1],[2,34,4],[1,34,4],[2,20,1]]
#frequence_y=0.8
#summary0 = summarize(dataset=dataset, columnWithClassResponse=0,frequence_y=frequence_y)
#summary1 = summarize(dataset=dataset, columnWithClassResponse=1,frequence_y=frequnce_y)
#summary_1 = summarize(dataset=dataset, columnWithClassResponse=-1,frequence_y=frequnce_y)
#print('sommaire 1 en enlevant col 0', summary0)
#print('sommaire 2 en enlevant col 1', summary1)
#print('sommaire 3 en enlevant col -1', summary_1)

# Sommaire par classe !
def summarizedByClass(dataset,columnWithClassResponse):
	separated = separateByClass(dataset=dataset,columnWithClassResponse=columnWithClassResponse)
	summaries = {}
	for classValue, instance in separated.items():
		# On ajoute la frequence d'apparition de la classe comme approximation de p(y)
		frequence_y=(len(instance) / len(dataset))
		summaries[classValue] = summarize(dataset=instance,columnWithClassResponse=columnWithClassResponse,frequence_y=frequence_y)
	return summaries

# Test summarizeByClass : IT WORKS
#dataset = [[1,20,1],[2,34,4],[1,34,4],[2,20,1],[1,24,4],[3,4,4],[3,4,4]]
#summaryByC0 = summarizedByClass(dataset=dataset, columnWithClassResponse=0)
#summaryByC1 = summarizedByClass(dataset=dataset, columnWithClassResponse=1)
#summaryByC_1 = summarizedByClass(dataset=dataset, columnWithClassResponse=-1)
#print('sommaire 1 par classe en col 0', summaryByC0)
#print('sommaire 2 par classe en col col 1', summaryByC1)
#print('sommaire 3 par classe en col col -1', summaryByC_1)



# 3 ) Make predictions : CAS NAIVE BAYES

#Calculate Gaussian Probability Density Function
#C'est cette fonction qui devra être changée et donc on fera la MAJ avec les fonctions de notre kernel !
import math
# Ici on devra faire 2 parties :
# une pour proba haute et une autre pour proba basse
#CALCUL PROBA CONDITIONNELLE
def calculateProbabilityNaiveBayes(x, mean, stdev2):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1/(math.sqrt(2*math.pi)*stdev)) *exponent

# Test calcul proba : IT WORKS !
#x = 71.5
#mean = 73
#stdev = 6.2
#frequence_y=0.5
#probability = calculateProbability(x=x,mean=mean,stdev=stdev,frequence_y=frequence_y)
#print('Proba de x = 71.5 d appartenir a la classe de moyenne 71.5 et stdev 6.2 selon Naive Bayes:',probability)

# Calcul des probas de classes maintenant !

'''
Now that we can calculate the probability of an attribute belonging to a class,
we can combine the probabilities of all of the attribute values for a data instance
and come up with a probability of the entire data instance belonging to the class.

We combine probabilities together by multiplying them. In the calculateClassProbabilities() below,
the probability of a given data instance is calculated by multiplying together
the attribute probabilities for each class. the result is a map of class values to probabilities.
'''

def calculateClassProbabilitiesNaiveBayes(summaries,inputVector): #,columnWithClassResponse): Pas important ici je pense
	probabilities = {}
	for classValue,classSummaries in summaries.items():
		probabilities[classValue] = 1 #Initialisation pour la multiplication ensuite des probas
		for i in range(len(classSummaries)):
			mean, stdev2, frequence_y = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbabilityNaiveBayes(x=x, mean=mean, stdev=stdev2)
			probabilities[classValue] *= frequence_y # multiplication par l'estimation de p(y)
	return probabilities

#Test proba par classe : IT WORKS
#summaries = {0:[(1, 0.5,.8)], 1:[(20, 5.0,.2)]}
#inputVector = [1.1] # pas besoin ici de mettre une fausse valeur en y ou même de faire inputVector = [1.1, ]
#probabilities = calculateClassProbabilities(summaries, inputVector)
#print('Probabilities for each class: ',probabilities)

# 4 ) Prédictions !

# Retourner 1 prediction :

def predictNaiveBayes(summaries, inputVector):
	probabilities = calculateClassProbabilitiesNaiveBayes(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

# Test prediction : IT WORKS

#summaries = {0:[(1, 0.5)], 1:[(20, 5.0)]}
#inputVector = [20.1] # pas besoin ici de mettre une fausse valeur en y ou même de faire inputVector = [1.1, ]
#prediction = predict(summaries, inputVector)
#print('Prediction de la classe class: ',prediction)

# Predictions sur un jeu de test complet :
def getPredictionsNaiveBayes(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predictNaiveBayes(summaries, testSet[i])
		predictions.append(result)
	return predictions

# Test : IT WORKS
#summaries = {'A':[(1, 0.5)], 'B':[(20, 5.0)]}
#testSet = [[1.1], [19.1],[18],[0]]
#predictions = getPredictions(summaries, testSet)
#print('Predictions: ',predictions)

# 5 ) Moyenne des erreurs :

def getAccuracyNaiveBayes(testSet, predictions, columnWithClassResponse):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][columnWithClassResponse] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

# Test :
#testSet = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
#predictions = ['a', 'a', 'a']
#accuracy = getAccuracy(testSet, predictions,3)
#print('Accuracy: ',accuracy)




# ADAPTATION D UNE PARTIE DU CODE POUR L UTILISATION DE NOTRE KERNEL IMPRECIS :
# 3 ) Make predictions :

# ON UTILISE LES FONCTIONS DU KERNEL PLUTOT QUE CELLE LA POUR NOS ESTIMATION

from statistics import stdev
from classes.Kernels.TriangularKernel import TriangularKernel
from classes.KernelContext import KernelContext
stepLinspace = 0.1
sigma=stdev(sample)



#Cette fonction est la MAJ avec les fonctions de notre kernel !
#import math
# Ici on devra faire 2 parties :
# une pour proba haute et une autre pour proba basse
#CALCUL PROBA CONDITIONNELLE
def calculateLowProbabilityImpreciseKernel(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1/(math.sqrt(2*math.pi)*stdev)) *exponent

#def calculateHightProbabilityImpreciseKernel(x, mean, stdev):
#	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
#	return (1/(math.sqrt(2*math.pi)*stdev)) *exponent

# Test calcul proba : IT WORKS !
#x = 71.5
#mean = 73
#stdev = 6.2
#frequence_y=0.5
#lowProbability = calculateLowProbability(x=x,mean=mean,stdev=stdev)
#hightProbability = calculateHightProbability(x=x,mean=mean,stdev=stdev)
#print('Proba de x = 71.5 d appartenir a la classe de moyenne 71.5 et stdev 6.2 selon Naive Bayes:',probability)

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

def calculateClassLowProbabilitiesImpreciseKernel(summaries,inputVector): #,columnWithClassResponse): Pas important ici je pense
	from statistics import stdev
	from classes.Kernels.TriangularKernel import TriangularKernel
	from classes.KernelContext import KernelContext
	stepLinspace = 0.1
	probabilities = {}
	for classValue,classSummaries in summaries.items():
		probabilities[classValue] = 1 #Initialisation pour la multiplication ensuite des probas
		for i in range(len(classSummaries)):
			mean, stdev, frequence_y = classSummaries[i]
			x = inputVector[i]
            lowProbabilities[classValue] *= calculateLowProbabilityImpreciseKernel(x=x, mean=mean, stdev=stdev)
			lowProbabilities[classValue] *= frequence_y # multiplication par l'estimation de p(y)
	return lowProbabilities

def calculateClassHightProbabilitiesImpreciseKernel(summaries,inputVector): #,columnWithClassResponse): Pas important ici je pense
	probabilities = {}
	for classValue,classSummaries in summaries.items():
		probabilities[classValue] = 1 #Initialisation pour la multiplication ensuite des probas
		for i in range(len(classSummaries)):
			mean, stdev, frequence_y = classSummaries[i]
			x = inputVector[i]
			hightProbabilities[classValue] *= calculatehightProbabilityImpreciseKernel(x=x, mean=mean, stdev=stdev)
			hightProbabilities[classValue] *= frequence_y # multiplication par l'estimation de p(y)
	return hightProbabilities

#Test proba par classe : IT WORKS
#summaries = {0:[(1, 0.5,.8)], 1:[(20, 5.0,.2)]}
#inputVector = [1.1] # pas besoin ici de mettre une fausse valeur en y ou même de faire inputVector = [1.1, ]
#probabilities = calculateClassProbabilities(summaries, inputVector)
#print('Probabilities for each class: ',probabilities)

# 4 ) Prédictions !

# Retourner 1 prediction avec 1 ou plusieurs classes :

def predictImpreciseKernel(summaries, inputVector):
	lowProbabilities = calculateClassLowProbabilitiesImpreciseKernel(summaries, inputVector)
	hightProbabilities = calculateClassHightProbabilitiesImpreciseKernel(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

# Test prediction : IT WORKS

#summaries = {0:[(1, 0.5)], 1:[(20, 5.0)]}
#inputVector = [20.1] # pas besoin ici de mettre une fausse valeur en y ou même de faire inputVector = [1.1, ]
#prediction = predict(summaries, inputVector)
#print('Prediction de la classe class: ',prediction)

# Predictions sur un jeu de test complet :
def getPredictionsImpreciseKernel(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predictNaiveBayes(summaries, testSet[i])
		predictions.append(result)
	return predictions

# Test : IT WORKS
#summaries = {'A':[(1, 0.5)], 'B':[(20, 5.0)]}
#testSet = [[1.1], [19.1],[18],[0]]
#predictions = getPredictions(summaries, testSet)
#print('Predictions: ',predictions)

# 5 ) Moyenne des erreurs :

def getAccuracyImpreciseKernel(testSet, predictions, columnWithClassResponse):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][columnWithClassResponse] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

# Test :
#testSet = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
#predictions = ['a', 'a', 'a']
#accuracy = getAccuracy(testSet, predictions,3)
#print('Accuracy: ',accuracy)





# CODE POUR LANCER LES FONCTIONS ET PREDIRE :

def main():
	file = 'iris.data.csv'
	splitRatio = 0.67
	dataset = loadCsv(file)
	trainingSet, testSet = splitDataset(dataset, splitRatio)
	# prepare model
	print('Split ',len(dataset),' rows into train=',len(trainingSet),' and test=',len(testSet),' rows')
	summaries = summarizedByClass(trainingSet,columnWithClassResponse=4)
	# test model
	predictionsNB = getPredictionsNaiveBayes(summaries, testSet)
	predictionsIK = getPredictionsNaiveBayes(summaries, testSet)
	accuracyNB = getAccuracyNaiveBayes(testSet, predictionsNB,(4))
	accuracyIK = getAccuracyNaiveBayes(testSet, predictionsIK,(4))
	print('Accuracy Naive Bayes : ',accuracyNB)
	print('Accuracy Naive Bayes : ',accuracyIK)

main()

# Tableaau des identifiants des réponses :
#0 = setosa
#1 = versicolor
#2 = virginica