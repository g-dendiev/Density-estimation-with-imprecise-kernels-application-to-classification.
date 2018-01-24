# NAIVE BAYE SFROM SCRATCH AVEC : https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/

# TEST DE CALCUL AVEC MAXIMALITE !


import matplotlib.pylab as plt
import numpy as np

from statistics import stdev

from classes.Kernels.TriangularKernel import TriangularKernel
from classes.KernelContext import KernelContext

# 1 ) On importe les données

import csv


def loadCsv(filename):
	lines = csv.reader(open(filename, "rt"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset


# Test import : IT WORKS

# filename = 'pima-indians-diabetes.data.csv'
# dataset = loadCsv(filename=filename)
# print('load data au format',filename, 'avec', len(dataset) ,'lignes ')

# Séparation des données

import random


def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while (len(trainSet) < trainSize):
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]


# Test split : IT WORKS

# dataset = [[1],[2],[3],[4],[5]]
# splitRatio = (2/3)
# train, test = splitDataset(dataset=dataset, splitRatio=splitRatio)
# print('on a split avec en test :',test,'et en train : ',train)


# 2 ) Sommaire des données

# Separation des données

# On ajoute une variable qui définit la colonne qui contient la réponse = la classe dans le dataset
def separateByClass(dataset, columnWithClassResponse):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		# On crée la ligne de la classe si elle existe pas
		if (vector[columnWithClassResponse] not in separated):
			separated[vector[columnWithClassResponse]] = []
		separated[vector[columnWithClassResponse]].append(vector)
	return separated


# Test separation des donnees : IT WORKS

# dataset = [[1,20,1],[2,34,4],[1,34,4],[2,20,1]]
# separated0 = separateByClass(dataset=dataset, columnWithClassResponse=0)
# separated1 = separateByClass(dataset=dataset, columnWithClassResponse=1)
# separated_1 = separateByClass(dataset=dataset, columnWithClassResponse=-1)
# print('separation 1 sur col 0', separated0)
# print('separation 2 sur col 1', separated1)
# print('separation 3 sur col -1', separated_1)

def separateFrequence(generalLowProbabilities,generalHightProbabilities):
	frequence_y = {}
	generalLowProbabilitiesWithoutFrequence = {}
	generalHightProbabilitiesWithoutFrequence = {}
	for classValue, tabResult in generalLowProbabilities.items():
		n = len(tabResult)
		frequence_y[classValue] = tabResult[-1] #frequence stockée en derniere position
		generalLowProbabilitiesWithoutFrequence[classValue] = []
		generalHightProbabilitiesWithoutFrequence[classValue] = []
		for i in (range(n-1)):
			generalLowProbabilitiesWithoutFrequence[classValue].append(tabResult[i])
			generalHightProbabilitiesWithoutFrequence[classValue].append(tabResult[i])
	return generalLowProbabilitiesWithoutFrequence,generalLowProbabilitiesWithoutFrequence, frequence_y

# Calcul de données de stats :

import math


def mean(numbers):
	return sum(numbers) / len(numbers)


def stdev2(numbers):
	avg = mean(numbers=numbers)
	variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
	return math.sqrt(variance)


# ADAPTATION D UNE PARTIE DU CODE POUR L UTILISATION DE NOTRE KERNEL IMPRECIS :
# 3 ) Make predictions :

# ON UTILISE LES FONCTIONS DU KERNEL PLUTOT QUE CELLE LA POUR NOS ESTIMATION

def InitHOptKernelImprecise(dataset):
	from statistics import stdev
	sigma = stdev(dataset)
	hOpt = 1.06 * sigma * (len(dataset)) ** (-1 / 5)
	return hOpt


# Cette fonction est la MAJ avec les fonctions de notre kernel !
# import math
# Ici on devra faire 2 parties :
# une pour proba haute et une autre pour proba basse

import math


def calculsDi(x, Xi):
	sumDi = 0
	# print('Xi = ',Xi)
	# print('x= ',x)
	for i in range(len(Xi)):
		sumDi += abs(x - Xi[i])
	meanDi = sumDi / len(Xi)
	return sumDi, meanDi


# CALCUL PROBA CONDITIONNELLE
# AJOUTER UN PARAMETRE EPSILON POUR EVITER LES PROBAS NULLES ICI C'EST FAIT A LA MAIN DANS FONCTION
def findProbabilityImpreciseKernel(generalLowProbabilities, generalHightProbabilities, tKernelDomain, x ,epsilon = 0.001):
	# BUT : créer un dictionnaire pour proba low et un pour proba hight et on renvoie à chaque fois
	# toutes ces données pour le vecteur d'étude passé en parametre !
	# But ensuite dans le predict : pouvoir comparer les probas hight et low des classes jointes ;)


	# Creation des dictionnaires
	localLowProbabilities = {}
	localHightProbabilities = {}
	tKernelDomainLow = {}
	tKernelDomainHight = {}
	tKernelDomainLow = tKernelDomain
	tKernelDomainHight = tKernelDomain

	#print('\n\n\n\ntKernelDomainLow dans le debut du find : ', tKernelDomainLow)
	#print('\n\n\n\ntKernelDomainHight dans le debut du find : ', tKernelDomainHight)

	frequence_y = {} # Dictionnaire avec les frequences des classes.
	generalLowProbabilitiesWithoutFrequence = {}
	generalHightProbabilitiesWithoutFrequence = {}

	generalLowProbabilitiesWithoutFrequence,generalHightProbabilitiesWithoutFrequence, frequence_y = separateFrequence(generalLowProbabilities,generalHightProbabilities)

	#print('generalLowProbabilitiesWithoutFrequence : ',generalLowProbabilitiesWithoutFrequence)
	#print('generalHightProbabilitiesWithoutFrequence : ', generalHightProbabilitiesWithoutFrequence)
	#print('frequence_y = ',frequence_y)


	# print('VALEUR DE X : ', x)
	# print(' PROBAS MAXIMALES :',generalHightProbabilities)
	# print(' PROBAS MINIMALES :', generalLowProbabilities)
	# print(' TKERNEL DOMAIN :', tKernelDomain)

	for classValue, tabResult in generalLowProbabilitiesWithoutFrequence.items():
		localLowProbabilities[classValue] = 1  # Initialisation pour la multiplication
		#print(' etat du domaine 1 : ', tKernelDomainLow)  # ATTENTION ON A UN TABLEAU A DEUX DIMENSION A GERER
		# 1ere DIMENSION = LA VAR UTILISEE, 2e = SA VALEUR AU POINT DU DOMAIN CALCULE !
		# DU COUP IL FAUT CHANGER LE FONCTIONNEMENT AVEC UN CHANGEMENT DE PARAMETRE ET D'APPEL
		# OU ALORS UNE GESTION DIFFERENTE AU SEIN DE LA FONCTION.
		#print('tabResult avant 2e boucle : ', tabResult)
		for i in range(len(tKernelDomainLow[classValue])):  # Iteration sur les differents attributs / var d'entree avec result specifique a  chaque classe
			# Initialisation de la valeur min du 2e indice
			jMin = 0
			#print(' i low = ', i)
			#print(' etat du domaine low 2 : ', tKernelDomainLow)
			#print('tKernelDomainlow de class value = ', classValue, ' est : ', tKernelDomainLow[classValue])
			#print('tKernelDomainlow de class value = ', classValue, ' pour le rang i =', i, ' est : ',tKernelDomainLow[classValue][i])
			# definition d'un domaine en lien avec la var en cours
			# definition d'un domaine en lien avec la var en cours
			tKernelDomainBis = []
			tKernelDomainBis.append(tKernelDomainLow[classValue][i])
			#print('\n low domain : ', tKernelDomainBis)
			#print('len(lowDomain) = ', len(tKernelDomainBis[0]))
			#print('\n Low domain : ', tKernelDomainBis)
			#print('len(lowDomain) = ',len(tKernelDomainBis[0]))

			#print(' domain bis avant boucle sur j : ',tKernelDomainBis)

			tableauVarClassEnCours = []
			tableauVarClassEnCours.append(tabResult[i])
			#print(' Entree 2e boucle, valeur de tabResult[i] : ',tabResult[i])
			#print(' valeur de tableauVarClassEnCours dans entree 2e boucle : ',tableauVarClassEnCours)
			#print(' Passage dans la deuxieme boucle du find. \n  len(tKernelDomainBis) = ',len(tKernelDomainBis[0]))
			#print('\n\n\n\ntKernelDomain avant passage dans for j dans low: ',tKernelDomainLow)  # print('x[i] =',x[i],'jMin low = ',jMin)
			for j in range(len(tKernelDomainLow[classValue][i])):
				#if j == 0:
				#	print(' etat du domaine low 3 : ', tKernelDomainLow)
				xMinTest = abs(tKernelDomainBis[0][j] - x[i])
				#if j == 0:
				#	print(' etat du domaine low 4 : ', tKernelDomainLow)
				#if i == 0:
				#	print('valeur xi = ',x[i])
				#	print(' APRES LOW Passage dans la 3e boucle du find.\n valeur de notre domaine en 0,',j,' = ',tKernelDomainBis[0][j],'\n\n')
				if (abs(tKernelDomainBis[0][jMin]-x[i]) > xMinTest):
					jMin = j
			# MAJ des probas conditionnelles
			# Ici on a un tableau qui contient un tbleau sur sa premiere ligne
			# On doit utiliser jMin dans le taleau de la 1ere ligne du tableau


			#print('\n\n\n\ntKernelDomain avant multiplication dans low: ',tKernelDomain)
			# print('x[i] =',x[i],'jMin low = ',jMin)
			if (tableauVarClassEnCours[0][jMin] == 0):
				print('valeur nulle on penalise par : ',epsilon)
				localLowProbabilities[classValue] *= epsilon  # On penalise par epsilon mais au moins on multiplie pas pasr 0 !
			else:
				localLowProbabilities[classValue] *= tableauVarClassEnCours[0][jMin]  # On penalise par epsilon mais au moins on multiplie pas pasr 0 !


			#print('\n\n\n\ntKernelDomain apres multiplication dans low: ',tKernelDomain)
		#print('\n\n\n\ntKernelDomain en sortie des  multiplication dans low: ', tKernelDomain)
		#print('x[i] =',x[i],'jMin low = ',jMin)
		#print(' local Low avant multiplication par frequence : ', localLowProbabilities)
		#print(' frequence de class en cours : ', frequence_y[classValue][0])
		# multiplication par la frequence et non le tableau contenant la frequence
		localLowProbabilities[classValue] *= frequence_y[classValue][0]


	#print('\n\n\n\ntKernelDomain avant passage dans boucle Higit: ', tKernelDomainHight)

	for classValue, tabResult in generalHightProbabilitiesWithoutFrequence.items():

		#print(' domain bis avant boucle sur j : ', tKernelDomainBis)

		localHightProbabilities[classValue] = 1  # Initialisation pour la multiplication

		for i in range(len(tabResult)):  # Iteration sur les differents attributs / var d'entree avec result specifique a  chaque classe
			# Initialisation de la valeur min du 2e indice
			jMin = 0
			#print(' i hight = ',i,'x[i] =',x[i])
			#print(' etat du domaine : ',tKernelDomainHight)
			#print('tKernelDomainHight de class value = ',classValue,' est : ',tKernelDomainHight[classValue])
			#print('tKernelDomainHight de class value = ', classValue, ' pour le rang i =',i,' est : ',tKernelDomainHight[classValue][i])  # definition d'un domaine en lien avec la var en cours
			tKernelDomainBis = []
			tKernelDomainBis.append(tKernelDomainHight[classValue][i])
			#print('\n Hight domain : ', tKernelDomainBis)
			#print('len(hightDomain) = ',len(tKernelDomainBis[0]))

			tableauVarClassEnCours = []
			tableauVarClassEnCours.append(tabResult[i])

			for j in range(len(tKernelDomainBis[0])):
				#if j == 0:
				#	print(' etat du domaine hight 3 : ', tKernelDomainHight)
				xMinTest = abs(tKernelDomainBis[0][j] - x[i])
				#if j == 0:
				#	print(' etat du domaine hight 4 : ', tKernelDomainHight)
				#if i == 0:
					#print(' AVANT HIGHT  Passage dans la 3e boucle du find.\n valeur de notre domaine en 0,', j, ' = ',tKernelDomainBis[0][j])
				#tKernelDomainBis[0][j] = abs(tKernelDomainBis[0][j] - x[i])
				#if i ==0:
				#	print('valeur xi = ', x[i])
				#	print(' APRES  HIGHT Passage dans la 3e boucle du find.\n valeur de notre domaine en 0,', j, ' = ',tKernelDomainBis[0][j], '\n\n')
				if (abs(tKernelDomainBis[0][jMin]-x[i]) > xMinTest):
					jMin = j

			#print('x[i] =',x[i],'jMin Hight = ', jMin)  # MAJ des probas conditionnelles
			if (tableauVarClassEnCours[0][jMin] == 0):
				localHightProbabilities[classValue] *= epsilon  # On penalise par epsilon mais au moins on multiplie pas pasr 0 !
			else:
				localHightProbabilities[classValue] *= tableauVarClassEnCours[0][jMin]  # On penalise par epsilon mais au moins on multiplie pas pasr 0 !
		#print(' local Hight avant multiplication par frequence : ',localHightProbabilities)
		# multiplication par la frequence et non le tableau contenant la frequence
		localHightProbabilities[classValue] *= frequence_y[classValue][0]

	#print(' FIN DU FIND on a : \n localLowProbabilities = ', localLowProbabilities,'\n localHightProbabilities = ', localHightProbabilities)

	return localLowProbabilities, localHightProbabilities


def calculateProbabilityImpreciseKernel(datasetTotalVar, datasetClass, h, epsilon, N):
	#Pour avoir les mêmes domaines sur toutes les classes, faire un premier kernelTri avec toutes les données de la var
	#Itérer ensuite sur ce domaine là mais avec nos data de classe !
	#Cela permet d'avoir un domain fixe ;)

	# Problème : avoir une taille de domaine qui soit la même pour chaque classe ! Non pas une valeur fixe !


	# IDEE : POUR POUVOIR FAIRE DES OPERATIONS SUR LES TABLEAUX, IL DOIVENT AVOIR LA MEME TAILLE !

	# Astuce pour jouer avec le updateDomain qui est dans le KenrelContext.
	# On met comme numérateur de self.stepLinspace le numérateur de la fraction qu détermine le nombre de points.
	# Comme ça notre dénominateur de stepLinspace devient le nombre de points (si on y ajoute 1 !)
	#Par exemple : si on divise par 9 notre stapLinspace, on aura donc 9+1 = 10 points dans le linspace !
	# Le domaine de def est défini par [min(Dataset) - h, max(Dataset) + h]
	minDomain = min(datasetTotalVar) - h
	maxDomain = max(datasetTotalVar) + h
	stepLinspace = (math.floor(maxDomain-minDomain))/20  # ATTENTION Mettre un plus grand dénominateur pour les phases de test



	lowProbabilities = []
	hightProbabilities = []
	generalDomain = []
	# print('DATASET AVANT PASSAGE DANS KERNEL :', dataset)
	tKernelTriGlobal = KernelContext(datasetTotalVar, TriangularKernel(h), stepLinspace)
	tKernelTriClass = KernelContext(datasetClass, TriangularKernel(h), stepLinspace)
	# print("Passage dans calculate Probability avec un nombre de points dans le domaine global qui vaut : ",len(tKernelTriGlobal.domain))
	for pt in tKernelTriGlobal.domain:
		# Def des structures qui vont récolter les données (dans la boucle pour une remise à 0 à chaque cycle
		# lenDomain.append(len(tKernelTri.domain))
		generalDomain.append(pt)
		structHMin = {'potentialHValue': -1, 'minValue': -1}

		structHMax = {'potentialHValue': -1, 'maxedValue': -1}

		# Calculs de f(hMax), et f(hMin)
		structHMax = tKernelTriClass.computeHMaxFromInterval(pt, h, epsilon)
		structHMin = tKernelTriClass.computeHMinFromInterval(pt, h, epsilon)

		hightProbabilities.append(structHMax['maxedValue'])
		lowProbabilities.append(structHMin['minValue'])
	# VOIR CE QUI VA OU QUI VA PAS CAR ON A LES MEMES VALEURS DANS LOW ER DANS HIGHT !
	#print('Low   probabilities = ', lowProbabilities)
	#print('Hight probabilities = ', hightProbabilities)

	return lowProbabilities, hightProbabilities, generalDomain


# Test calcul proba : IT WORKS !
# x = 71.5
# mean = 73
# stdev = 6.2
# frequence_y=0.5
# lowProbability = calculateLowProbability(x=x,mean=mean,stdev=stdev)
# hightProbability = calculateHightProbability(x=x,mean=mean,stdev=stdev)
# print('Proba de x = 71.5 d appartenir a la classe de moyenne 71.5 et stdev 6.2 selon Naive Bayes:',probability)


# Separation des classes en enlevant la colonne de la réponse:
# On ajoute une variable qui définit la colonne qui contient la réponse = la classe dans le dataset
# columnWithClassResponse doit être la premiere colonne (0) ou la dernière (-1)
def separateByClassWithoutResponse(dataset, columnWithClassResponse):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		# On crée la ligne de la classe si elle existe pas
		if (vector[columnWithClassResponse] not in separated):
			separated[vector[columnWithClassResponse]] = []
		if (columnWithClassResponse == -1):
			separated[vector[columnWithClassResponse]].append(
				vector[0:columnWithClassResponse])  # On supprime la var avec la classe
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

def calculateClassProbabilitiesImpreciseKernel(dataset, columnWithClassResponse, margeEpsilon):
	N = len(dataset)
	separated = separateByClassWithoutResponse(dataset, columnWithClassResponse)
	dataset2 = []
	hOpt = []
	for i in range(len(dataset)):
		if (columnWithClassResponse == -1):
			dataset2.append(dataset[i][0:columnWithClassResponse])
		else:
			dataset2.append(dataset[i][1:])
	dataset2Separated = [attribute1 for attribute1 in zip(*dataset2)]
	for i in range(len(dataset2Separated)):
		hOpt.append(InitHOptKernelImprecise(dataset2Separated[i]))
	# print(len(separated.items()))
	lowProbabilities = {}
	hightProbabilities = {}
	generalDomain = {}
	for classValue, classDataset in separated.items():
		# print('classValue :',classValue)
		# print('len classDataset',len(classDataset))
		# Separation des colonnes pour qu'une colonne corresponde à une seule variable d'entrée
		# print('zip : ',zip(*classDataset))
		classDatasetWithColSeparated = [attribute for attribute in zip(*classDataset)]
		# print('class : ',classDatasetWithColSeparated)
		# print(len(classDatasetWithColSeparated))
		# Initialisation pour pouvoir stocker des tableaux selon le nombre de variables en entrée
		lowProbabilities[classValue] = []
		hightProbabilities[classValue] = []
		generalDomain[classValue] = []

		# Initialisation de la frequence de la classe : estimateur de p(Y)
		frequence_y = len(classDataset) / N

		# Si on a au moins une var d'entree, alors on lance la machine. Sinon message d'erreur.
		if(len(classDatasetWithColSeparated) != 0):
			for i in range(len(classDatasetWithColSeparated)):
				# print(i)
				# Ici le vecteur d'entrée ne sert plus, on peut donc l'enlever de notre boucle de calcul.
				# x = inputVector[i]
				# print(x)
				# Calcul toute la densité sur la classe et ensuite on assigne les prbas selon les points en entrée et en sortie
				generalLowProbabilities, generalHightProbabilities, generalDomainI = calculateProbabilityImpreciseKernel(datasetTotalVar=dataset2Separated[i],datasetClass=classDatasetWithColSeparated[i], h=hOpt[i], epsilon=margeEpsilon * hOpt[i], N=N)

				###### ATTENTION ATTENTION ATTENTION
				# On peut pas procéder comme suit, il faut un tableau à 3 dimensions : class, n*var correspondants aux entrées !
				# if i == 0:
					#Initialisation de chaque sous tableau de classe
				#    lowProbabilities[classValue] = generalLowProbabilities
				# On cherchera les probas dans la phase où on predira la classe,
				#Le but ici est de retourner les tableaux complets de taille égale :) D'ou la mise en commentaire en dessous
				# lowProbability, hightProbability = findProbabilityImpreciseKernel(x, generalHightProbabilities,generalDomain, generalLowProbabilities)
				# else :
					#Ici on fait le produit membre à membre de chaque tableau de chaque classe
					#  pour les diférentes var au sein du système
					# Le but étant d'avoir les tableaux de
				#    lowProbabilities[classValue] *= lowProbability
				#    hightProbabilities[classValue] *= hightProbability

				##### IL FAUTDRA DONC PROCEDER COMME SUIT !
				# Créer un tableau avec chaque classe et les données pour chaque var !
				# Il faudra donc faire le produit de toutes les probas basses dans la fonction predict !
				# print('lowProbabilities = ',lowProbabilities)
				lowProbabilities[classValue].append(generalLowProbabilities)
				hightProbabilities[classValue].append(generalHightProbabilities)
				generalDomain[classValue].append(generalDomainI)

			# On stocke en dernier sous tableau la frequence d'apparition de la classe !
			lowProbabilities[classValue].append([frequence_y])
			hightProbabilities[classValue].append([frequence_y])
		else :
			print('ERREUR : IL N\'EXISTE AUCUNE VARIABLE POUR PRESIRE !!!!!!')
			break

	# print(' frequence d apparition de la Classe value = ',classValue,'est de :',frequence_y,', avec une Probabilité haute = ',hightProbabilities[classValue],', et une probabilité basse :', lowProbabilities[classValue])
	# ATTENTION prévoir une sorti avec le domaine quand on appel la fonction.
	#print('lowProbabilities = ',lowProbabilities)
	#print('hightProbabilities = ', hightProbabilities)

	return lowProbabilities, hightProbabilities, generalDomain


# testLow=calculateClassLowProbabilitiesImpreciseKernel(dataset=[[1,3,2,5],[2,6,3,5],[1,4,4,5]],columnWithClassResponse=0,inputVector=[4,7,7],hOpt=10,margeEpsilon=0.5)
# print('testLow = ',testLow)

# 4 ) Prédictions !

# Retourner 1 prediction avec 1 ou plusieurs classes :


###### CHANGEMENT A PREVOIR POUR FAIRE LA MAXIMALITE AVEC EN PARAMETRE LA SORTIE DU FIND PROBAS

def predictImpreciseKernel(lowProbabilities, hightProbabilities, generalDomain, inputVector, margeEpsilon):
	# calculateClassProbabilitiesImpreciseKernel(dataset,columnWithClassResponse,inputVector,margeEpsilon):
	bestLabel = []
	classLabel = []
	dominatedClass = []


	# VOIR POUR UTILISER LA FONCTION FIND PROBABILITIES :)


	localLowProbabilities = {}
	localHightProbabilities = {}

	localLowProbabilities, localHightProbabilities = findProbabilityImpreciseKernel(lowProbabilities, hightProbabilities, generalDomain, inputVector)


	# print('prediction de low probabilities : ',lowProbabilities)
	for classValueLow, lowProba in localLowProbabilities.items():
		classLabel.append(classValueLow)
		#print('class Value = ',classValueLow)
		#print('lowProba =',lowProba)
		#tjrsSup = 1
		if classValueLow not in dominatedClass:
			for classValueHight, hightProba in localHightProbabilities.items():
				if classValueHight not in dominatedClass:
					if classValueHight != classValueLow:
						# print('hightProba = ',hightProba)
						if (lowProba/hightProba) > 1 :
							dominatedClass.append(classValueHight)

	for cl in classLabel:
		if cl not in dominatedClass:
			bestLabel.append(cl)

	return bestLabel


# Test prediction : IT WORKS

# testPredict=predictImpreciseKernel(dataset=[[1,3,2,5],[2,6,3,5],[1,3.5,2.2,5.1],[3,3,2,5],[3,3.5,2.2,5.1]],columnWithClassResponse=0,inputVector=[3.5,2.8,6],hOpt=5,margeEpsilon=0.1)
# print('testpredict = ',testPredict)

### CHANGEmMENT A FAIRE :
# faire en sorte que le predict prenne en parametre:
#           les dictionnaires low et hight + generalDomain
# Ensuite on fait appel au calcul des probas hautes et basses par classe puis
# On appel predict sur chaque point de vontre vecteur d'entrée : A METTRE EN PARAMETRE LORS DE L'APPEL
# Le predict va utiliser le find( point d'entrée) et va nous retourner
#  les probas hautes et basses jointes de cahque classe en fonction de notre point d'etude.

# Predictions sur un jeu de test complet :
def getPredictionsImpreciseKernel(dataset, columnWithClassResponse, testSet, margeEpsilon):
	predictions = []
	# print('test set passé en argument =',testSet)
	lowProbabilities, hightProbabilities, generalDomain = calculateClassProbabilitiesImpreciseKernel(dataset,columnWithClassResponse,margeEpsilon)

	for i in range(len(testSet)):
		# print('test set de ',i,' = ',testSet[i])
		result = predictImpreciseKernel(lowProbabilities, hightProbabilities, generalDomain, testSet[i], margeEpsilon)
		# print('resultat pour la ligne ',i,' : ',result)
		predictions.append(result)
	return predictions


# Test : IT WORKS
# testSet=[[3.5,2.8,6],[3,3,5],[10.1,12.1,14.1]]
# testPredictions=getPredictionsImpreciseKernel(dataset=[[1,3,2,5],[2,10,12,14],[1,3.5,2.2,5.1],[3,3,2,5],[3,3.5,2.2,5.1]],columnWithClassResponse=0,testSet=testSet,hOpt=5,margeEpsilon=0.1)
# print('testpredictions = ',testPredictions)

# 5 ) Moyenne des erreurs :

def getAccuracyImpreciseKernel(testSet, predictions, columnWithClassResponse):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][columnWithClassResponse] in predictions[x]:
			correct += 1  # /len(predictions[x])
	return (correct / float(len(testSet))) * 100.0


# Test :
# testSet = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
# predictions = ['a', 'a', ['a','b']]
# accuracy = getAccuracyImpreciseKernel(testSet, predictions,3)
# print('Accuracy: ',accuracy)





# CODE POUR LANCER LES FONCTIONS ET PREDIRE :

def main():
	file = 'iris.data.csv'
	splitRatio = 0.20
	dataset = loadCsv(file)
	trainingSet, testSet = splitDataset(dataset, splitRatio)
	# prepare model
	print('Split ', len(dataset), ' rows into train=', len(trainingSet), ' and test=', len(testSet), ' rows')
	# summaries = summarizedByClass(trainingSet,columnWithClassResponse=4)
	# print('testSet = ',testSet)
	testSet2 = []
	valeurAttendue = []
	for i in range(len(testSet)):
		testSet2.append(testSet[i][0:4])
		valeurAttendue.append([testSet[i][4]])
	# test model
	predictionsPK = getPredictionsImpreciseKernel(dataset, columnWithClassResponse=-1, testSet=testSet2, margeEpsilon=0)
	# La colonne avec la réponse de classe doit être 0 ou -1 (1ere ou dernière colonne du dataset passé en parametre
	predictionsIK = getPredictionsImpreciseKernel(dataset, columnWithClassResponse=-1, testSet=testSet2, margeEpsilon=0.1)
	print('predictions PK =', predictionsPK)
	print('valeur atendue :', valeurAttendue)
	print('predictions IK =', predictionsIK)  # le zip permet d'enlever les sous-tableaux quand on a 1 seule valeur
	accuracyPK = getAccuracyImpreciseKernel(testSet, predictionsPK, (4))
	accuracyIK = getAccuracyImpreciseKernel(testSet, predictionsIK, (4))
	print('Accuracy Precise Kernel : ', accuracyPK)
	print('Accuracy Imprecise Kernel : ', accuracyIK)


main()

def launchXTimes(times):
	for i in range(times):
		# random.seed(i)
		print('Resultats de l\'iteration : ', i + 1)
		main()


#launchXTimes(1)


# Tableaau des identifiants des réponses :
# 0 = setosa
# 1 = versicolor
# 2 = virginica
