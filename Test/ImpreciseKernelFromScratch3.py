# NAIVE BAYE SFROM SCRATCH AVEC : https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/

# Implementation de la maximalite pour comparer les classes entre elles

# 0) Import des outils utiles des implementations du kernel triangulaire.

from classes.Kernels.TriangularKernel import TriangularKernel
from classes.KernelContext import KernelContext


# 1 ) On importe les données a format csv
# Les donnees qualitatives sont transformees en donnees quantitatives
# On utilise un dictionnaire pour faire le transfert.

import csv

def loadCsv(filename):
	lines = csv.reader(open(filename, "rt"))
	dataset = list(lines)
	dictionnaire = {}
	dictKey = 0
	for i in range(len(dataset)):
		for j in range(len(dataset[i])):
			try: # Conversion en float
				dataset[i][j] = float(dataset[i][j])
			except: # Conversion en int en passant par un dico clef valeur
				# Passage quantitatif a qualitatif !
				if dataset[i][j] not in dictionnaire.keys():
					dictionnaire[dataset[i][j]] = dictKey
					dictKey += 1
					dataset[i][j] = dictionnaire[dataset[i][j]]
				else:
					dataset[i][j] = dictionnaire[dataset[i][j]]
	return dataset


# 2) Séparation des données

import random

def splitDataset(dataset, splitRatio):
	if splitRatio > 1 or splitRatio < 0 :
		print('Erreur, le ratio de separation du jeu de donnee doit etre dans ]0,1[')
		return 0
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while (len(trainSet) < trainSize):
		# On met dans trainSet que le ratio voulu de donnees.
		# On enleve en meme temps ces donnees de la copie du dataset initial.
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

# 3) Separation des donnees d'apprentissage par classe

# On ajoute une variable qui définit la colonne qui contient la réponse = la classe dans le dataset
# columnWithClassResponse doit être la premiere colonne (0) ou la dernière (-1)
def separateByClass(dataset, columnWithClassResponse):
	if columnWithClassResponse not in (0,-1):
		print('Erreur, la colonne avec les reponses doit etre la premiere (0) ou la derniere (-1)')
		return 0
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		# On crée la ligne de la classe si elle existe pas
		if (vector[columnWithClassResponse] not in separated):
			separated[vector[columnWithClassResponse]] = []
		separated[vector[columnWithClassResponse]].append(vector)
	return separated


# 4) Initialisation du hOpt pour un dataset donnee

# Note : on passe en parametre toutes les donnees d'une var.
# On fera donc n appels a cette fonction pour n var differentes.
# Le but etant d'avoir un hOpt avec toutes les donnees avant de faire la regression classe par classe.

from statistics import stdev

def InitHOptKernelImprecise(dataset):
	if len(dataset) == 0:
		print('Erreur le dataset est vide pour l\'initialisation de hOpt')
		return 0
	sigma = stdev(dataset)
	hOpt = 1.06 * sigma * (len(dataset)) ** (-1 / 5)
	return hOpt

# 5) Division du dataset de test en enlevant la colonne de la réponse:

# On ajoute une variable qui définit la colonne qui contient la réponse = la classe dans le dataset
# columnWithClassResponse doit être la premiere colonne (0) ou la dernière (-1)
def separateByClassWithoutResponse(dataset, columnWithClassResponse):
	if columnWithClassResponse not in (0,-1):
		print('Erreur, la colonne avec les reponses doit etre la premiere (0) ou la derniere (-1)')
		return 0
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		# On crée la ligne de la classe si elle existe pas
		if (vector[columnWithClassResponse] not in separated):
			separated[vector[columnWithClassResponse]] = []
		if (columnWithClassResponse == -1):
			# On supprime la var avec la classe en derniere colonne
			separated[vector[columnWithClassResponse]].append(
				vector[0:columnWithClassResponse])
		else:
			# On supprime la var avec la classe en premiere colonne
			separated[vector[columnWithClassResponse]].append(vector[1:])
	return separated




# 6) On enleve la frequence presente en derniere colonne des calculs issus de nos regressions

def separateFrequence(generalLowProbabilities,generalHightProbabilities):
	frequence_y = {}
	generalLowProbabilitiesWithoutFrequence = {}
	generalHightProbabilitiesWithoutFrequence = {}
	for classValue, tabResult in generalLowProbabilities.items():
		n = len(tabResult)
		frequence_y[classValue] = tabResult[-1] #frequence stockée en derniere position
		generalLowProbabilitiesWithoutFrequence[classValue] = []
		generalHightProbabilitiesWithoutFrequence[classValue] = []
		for i in (range(n-1)): # On recopie tout sauf la deniere colonne
			generalLowProbabilitiesWithoutFrequence[classValue].append(tabResult[i])
			generalHightProbabilitiesWithoutFrequence[classValue].append(tabResult[i])
	return generalLowProbabilitiesWithoutFrequence,generalLowProbabilitiesWithoutFrequence, frequence_y





# 7) Calculs lies a l'utilisation de nos implementations des kernels
import math

def calculsDi(x, Xi):
	sumDi = 0
	for i in range(len(Xi)):
		sumDi += abs(x - Xi[i])
	meanDi = sumDi / len(Xi)
	return sumDi, meanDi



# 8) Calcul des probas conditionnelles jointes (hautes et basses) pour appliquer la maximalite ensuite
#  Ajout d'un parametre de penalite afin de penaliser notre produit de probas si une des probas rencontrees est nulle.
def findProbabilityImpreciseKernel(generalLowProbabilities, generalHightProbabilities, tKernelDomain, inputVector,penalite = 0.001):

	if (len(generalLowProbabilities) == 0 or len(generalHightProbabilities) == 0 or len(tKernelDomain) == 0 or len(inputVector) == 0):
		print('Erreur dans un des parametre du findProbabilityImpreciseKernel')
		if (len(generalLowProbabilities) == 0):
			print('le parametre -generalLowProbabilities- est vide')
		if (len(generalHightProbabilities) == 0):
			print('le parametre -generalHightProbabilities- est vide')
		if (len(tKernelDomain) == 0):
			print('le domaine -tKernelDomain- est vide')
		if(len(inputVector) == 0):
			print('le vecteur d\'entree -inputVector- est vide')

	# Creation des dictionnaires
	## stockage des resultats classe par classe du vecteur x passe en parametre
	localLowProbabilities = {}
	localHightProbabilities = {}
	## copie des probas generale sans la frequence, et dict pour la frequence par classe
	generalLowProbabilitiesWithoutFrequence = {}
	generalHightProbabilitiesWithoutFrequence = {}
	frequence_y = {}

	# Copie du domaine
	tKernelDomainLow = {}
	tKernelDomainHight = {}
	tKernelDomainLow = tKernelDomain
	tKernelDomainHight = tKernelDomain

	# Separation des calculs lies a la regression et de la frequence
	generalLowProbabilitiesWithoutFrequence,generalHightProbabilitiesWithoutFrequence, frequence_y = separateFrequence(generalLowProbabilities,generalHightProbabilities)

	# Partie probas basses :
	# Boucle sur les classes du dictionnaire et leur regression pour chaque var que l'on etudie
	for classValue, tabResult in generalLowProbabilitiesWithoutFrequence.items():
		# Initialisation pour la multiplication
		localLowProbabilities[classValue] = 1

		# Attention on a un tableau de resultats a plusieurs dimensions a gerer
		# 1ere dimension = la var utilisee
		# 2e = la valeur de la regression au point d'application du domaine de regression

		# Iteration sur les differents attributs / var d'entree avec des resultats specifiques a chaque classe
		for i in range(len(tKernelDomainLow[classValue])):
			# Initialisation de l'indice ou notre x est le plus proche d'un point ou l'on a fait la regression
			jMin = 0

			# definition d'un domaine en lien avec la var en cours
			tKernelDomainBis = []
			tKernelDomainBis.append(tKernelDomainLow[classValue][i])

			# On stocke les resultats de la regression de la classe en cours sur la var en cours
			tableauVarClassEnCours = []
			tableauVarClassEnCours.append(tabResult[i])

			# On itere sur une var d'un dictionnaire contenant un tableau a 2 dimensions (cf avant)
			# But : trouver le point pour lequel on est le plus proche du point etudie (x[i]) du vecteur x en entree
			for j in range(len(tKernelDomainLow[classValue][i])):
				xMinTest = abs(tKernelDomainBis[0][j] - inputVector[i])
				if (abs(tKernelDomainBis[0][jMin]-inputVector[i]) > xMinTest):
					jMin = j

			# MAJ des probas conditionnelles
			# Ici on a un tableau qui contient un tbleau sur sa premiere ligne
			# On doit utiliser jMin dans le tableau de la 1ere ligne du tableau
			if (tableauVarClassEnCours[0][jMin] == 0):
				# On penalise par un petit coef mais au moins on multiplie pas par 0 !
				#print('valeur nulle en probabilite basse au point d\'etude ',inputVector[i],', du vecteur : ',inputVector,', on penalise par : ',penalite)
				localLowProbabilities[classValue] *= penalite
			else:
				# On multiplie nos probas de classe par la probas de classe de la var en cours
				localLowProbabilities[classValue] *= tableauVarClassEnCours[0][jMin]

		# multiplication par la frequence et non par le tableau contenant la frequence pour la classe en cours
		localLowProbabilities[classValue] *= frequence_y[classValue][0]

	# Partie probas basses :
	# Boucle sur les classes du dictionnaire et leur regression pour chaque var que l'on etudie
	for classValue, tabResult in generalHightProbabilitiesWithoutFrequence.items():
		# Initialisation pour la multiplication
		localHightProbabilities[classValue] = 1

		# Attention on a un tableau de resultats a plusieurs dimensions a gerer
		# 1ere dimension = la var utilisee
		# 2e = la valeur de la regression au point d'application du domaine de regression

		# Iteration sur les differents attributs / var d'entree avec des resultats specifiques a chaque classe
		for i in range(len(tabResult)):
			# Initialisation de l'indice ou notre x est le plus proche d'un point ou l'on a fait la regression
			# Remarque : on a le meme domaine que pour les probas basses
			# Donc pour un meme vecteur x en entree on aura en basse et en haute les memes points d'application de la
			# regression. Ce qui permet d'avoir les probas jointes hautes et basses conditionnellement aux classes.
			jMin = 0

			# definition d'un domaine en lien avec la var en cours
			tKernelDomainBis = []
			tKernelDomainBis.append(tKernelDomainHight[classValue][i])

			# On stocke les resultats de la regression de la classe en cours sur la var en cours
			tableauVarClassEnCours = []
			tableauVarClassEnCours.append(tabResult[i])

			# On itere sur une var d'un dictionnaire contenant un tableau a 2 dimensions (cf avant)
			# But : trouver le point pour lequel on est le plus proche du point etudie (x[i]) du vecteur x en entree
			# Note : on a le meme domaine que pour les probas basses donc on aura le meme indice de tableau en sortie
			# C'est ce qui permet d'avoir les probas jointes hautes et basses conditionnellement aux classes
			for j in range(len(tKernelDomainBis[0])):
				xMinTest = abs(tKernelDomainBis[0][j] - inputVector[i])
				if (abs(tKernelDomainBis[0][jMin]-inputVector[i]) > xMinTest):
					jMin = j

			# MAJ des probas conditionnelles
			# Ici on a un tableau qui contient un tbleau sur sa premiere ligne
			# On doit utiliser jMin dans le tableau de la 1ere ligne du tableau
			if (tableauVarClassEnCours[0][jMin] == 0):
				# On penalise par un petit coef mais au moins on multiplie pas par 0 !
				#print('valeur nulle en probabilite haute au point d\'etude ',inputVector[i],', du vecteur : ',inputVector,', on penalise par : ', penalite)
				localHightProbabilities[classValue] *= penalite
			else:
				# On multiplie nos probas de classe par la probas de classe de la var en cours
				localHightProbabilities[classValue] *= tableauVarClassEnCours[0][jMin]
		# multiplication par la frequence et non par le tableau contenant la frequence pour la classe en cours
		localHightProbabilities[classValue] *= frequence_y[classValue][0]

	return localLowProbabilities, localHightProbabilities


# 9) Calcul des pobabilites pour une classe donnee en parametre
def calculateProbabilityImpreciseKernel(datasetTotalVar, datasetClass, h, epsilon, nbPointsRegressionMoins1 = 99):
	if (len(datasetTotalVar) == 0 or len(datasetClass) == 0 or h == 0 or nbPointsRegressionMoins1 < 0):
		print('Erreur dans un des parametre du calculateProbabilityImpreciseKernel')
		if (len(datasetTotalVar) == 0):
			print('le parametre -datasetTotalVar- est vide')
		if (len(datasetClass) == 0):
			print('le parametre -datasetClass- est vide, une cclasse en trop a ete trouvee')
		if (h == 0):
			print('le parametre -h- est null, pas de regression possible')
		if (nbPointsRegressionMoins1 < 0):
			print('le nombre de poins pour la regression -nbPointsRegressionMoins1- doit etre positif')

	# Pour avoir les mêmes domaines sur toutes les classes, faire un premier kernelTri avec toutes les données de la var
	# On itére ensuite sur ce domaine là mais avec nos donnees de classe !
	# Cela permet d'avoir un domain fixe !
	# De plus une astuce un parametre permet de definir un nombre de points pour notre regressions.
	# Note : pour avoir 40 points dans la regression, mettre 39 en parametre.

	# ATTENTION : POUR POUVOIR FAIRE DES OPERATIONS SUR LES TABLEAUX ENSUITE, IL DOIVENT AVOIR LA MEME TAILLE !
	# D'ou l'utilisation de la variable nbPointsRegression

	# Astuce pour jouer avec le updateDomain qui est dans le KenrelContext.
	# On met comme numérateur de self.stepLinspace le numérateur de la fraction qu détermine le nombre de points.
	# Comme ça notre dénominateur de stepLinspace devient le nombre de points (si on y ajoute 1 !)
	# Par exemple : si on divise par 9 notre stapLinspace, on aura donc 9+1 = 10 points dans le linspace !
	# Le domaine de def est défini par [min(Dataset) - h, max(Dataset) + h]
	minDomain = min(datasetTotalVar) - h
	maxDomain = max(datasetTotalVar) + h
	stepLinspace = (math.floor(maxDomain-minDomain))/nbPointsRegressionMoins1

	# Creation des tableaux contenant les resultats pour la var en cours et la classe en cours
	lowProbabilities = []
	hightProbabilities = []
	generalDomain = []

	# Dictinction regression sur tout le dataset et sur le dataset de la classe en cours
	tKernelTriGlobal = KernelContext(datasetTotalVar, TriangularKernel(h), stepLinspace)
	tKernelTriClass = KernelContext(datasetClass, TriangularKernel(h), stepLinspace)

	# Calcul de nos probas min et max en chaque points de la regression pour la classe en cours
	for pt in tKernelTriGlobal.domain:

		# Stockage du domaine en cours
		generalDomain.append(pt)

		# Def des structures qui vont récolter les données
		# (def dans la boucle pour une remise à 0 à chaque cycle)
		structHMin = {'potentialHValue': -1, 'minValue': -1}
		structHMax = {'potentialHValue': -1, 'maxedValue': -1}

		# Calculs de f(hMax), et f(hMin)
		structHMax = tKernelTriClass.computeHMaxFromInterval(pt, h, epsilon)
		structHMin = tKernelTriClass.computeHMinFromInterval(pt, h, epsilon)

		# Stockage successifs des resultats
		hightProbabilities.append(structHMax['maxedValue'])
		lowProbabilities.append(structHMin['minValue'])

	# On retourne egalement le domaine pour pouvoir ensuite faire les probas jointes hautes et basses.
	return lowProbabilities, hightProbabilities, generalDomain


# 10 ) Fonction qui va etre appelee une fois pour calculer les probas imprecis de chaque var pour chaque classe.
# La marge est celle passée dans la fonction principale en parametre, elles est en pourcentage.
def calculateClassProbabilitiesImpreciseKernel(dataset, columnWithClassResponse, margeEpsilon):
	# Declaration / initialisation des variables
	N = len(dataset)
	dataset2 = []
	hOpt = []
	lowProbabilities = {}
	hightProbabilities = {}
	generalDomain = {}

	# Separation des classes sans la colonne de reponse
	separated = separateByClassWithoutResponse(dataset, columnWithClassResponse)


# copie du dataset sans la colonne contenant la reponse
	for i in range(len(dataset)):
		if (columnWithClassResponse == -1):
			dataset2.append(dataset[i][0:columnWithClassResponse])
		else:
			dataset2.append(dataset[i][1:])

	# Separation des colonnes avec la fonction zip
	dataset2Separated = [attribute1 for attribute1 in zip(*dataset2)]

	#On initialisae nos hOpt pour chaque var
	for i in range(len(dataset2Separated)):
		hOpt.append(InitHOptKernelImprecise(dataset2Separated[i]))

	# Pour chaque classe et chaque variable on fait nos calculs
	for classValue, classDataset in separated.items():
		# Separation des colonnes pour qu'une colonne corresponde à une seule variable d'entrée
		classDatasetWithColSeparated = [attribute for attribute in zip(*classDataset)]

		# Initialisation pour pouvoir stocker des tableaux selon le nombre de variables en entrée
		lowProbabilities[classValue] = []
		hightProbabilities[classValue] = []
		generalDomain[classValue] = []

		# Initialisation de la frequence de la classe : estimateur de p(Y)
		frequence_y = len(classDataset) / N

		# Si on a au moins une var d'entree, alors on lance la machine. Sinon message d'erreur.
		if(len(classDatasetWithColSeparated) != 0):
			for i in range(len(classDatasetWithColSeparated)):
				# Calcul toute la densité sur la classe et ensuite on assigne les prbas selon les points en entrée et en sortie
				generalLowProbabilities, generalHightProbabilities, generalDomainI = calculateProbabilityImpreciseKernel(datasetTotalVar=dataset2Separated[i],datasetClass=classDatasetWithColSeparated[i], h=hOpt[i], epsilon=margeEpsilon * hOpt[i])

				# On crée un tableau avec chaque classe et les données pour chaque var !
				# On fait donc le produit de toutes les probas basses dans la fonction predict !
				lowProbabilities[classValue].append(generalLowProbabilities)
				hightProbabilities[classValue].append(generalHightProbabilities)
				generalDomain[classValue].append(generalDomainI)

			# On stocke en dernier sous tableau la frequence d'apparition de la classe !
			lowProbabilities[classValue].append([frequence_y])
			hightProbabilities[classValue].append([frequence_y])
		else :
			print('ERREUR : IL N\'EXISTE AUCUNE VARIABLE POUR PREDIRE !!!!!!')
			break
	return lowProbabilities, hightProbabilities, generalDomain


# 11) On definit la fonction de prediction de classe avec la maximalite comme critere de comparaison
def predictImpreciseKernel(lowProbabilities, hightProbabilities, generalDomain, inputVector, margeEpsilon):
	# Declaration des variables
	bestLabel = []
	classLabel = []
	dominatedClass = []
	localLowProbabilities = {}
	localHightProbabilities = {}

	# Initialisation
	localLowProbabilities, localHightProbabilities = findProbabilityImpreciseKernel(lowProbabilities, hightProbabilities, generalDomain, inputVector)

	# On itere sur l'ensemble des classes du tableau de probas basses qui ne sont pas dominees
	for classValueLow, lowProba in localLowProbabilities.items():
		# On stocke tous les labels de classe
		classLabel.append(classValueLow)

		# On verifie que la classe en cours n'est pas dominee
		if classValueLow not in dominatedClass:

			# Boucle de comparaison avec les probas hautes des autres classes non dominees
			for classValueHight, hightProba in localHightProbabilities.items():
				if classValueHight not in dominatedClass:
					# On verifie qu'on ne compare pas les probas hautes et basses de la meme classe
					if classValueHight != classValueLow:
						# Critere de maximalite ! Si la classe Low / classe Hight > 1
						# alors la classe Hight est dominee. On l'ajoute aux classes dominees
						if (lowProba/hightProba) > 1 :
							dominatedClass.append(classValueHight)

	# Tous les labels non domines correspondent a la ou les classes possibles
	for cl in classLabel:
		if cl not in dominatedClass:
			bestLabel.append(cl)

	# On retourne le sous ensemble des labels possibles.
	return bestLabel


# 12) Predictions sur un jeu de test complet :
def getPredictionsImpreciseKernel(dataset, columnWithClassResponse, testSet, margeEpsilon):
	# Declaration
	predictions = []

	# Initialisation des probas hautes et basses, ainsi que du domaine de chaque variable
	lowProbabilities, hightProbabilities, generalDomain = calculateClassProbabilitiesImpreciseKernel(dataset,columnWithClassResponse,margeEpsilon)

	# Calcul de prediction pour chaque donnee de test
	for i in range(len(testSet)):
		result = predictImpreciseKernel(lowProbabilities, hightProbabilities, generalDomain, testSet[i], margeEpsilon)
		predictions.append(result)

	# On retourne le taleau complet des predictions
	return predictions


# 13 ) Moyenne des erreurs :
def getAccuracyImpreciseKernel(testSet, predictions, columnWithClassResponse):
	correct = 0
	for x in range(len(testSet)):
		# Penalite imprecise a ajouter
		if testSet[x][columnWithClassResponse] in predictions[x]:
			correct += 1  # insertion penalite
	return (correct / float(len(testSet))) * 100.0


# CODE POUR LANCER LES FONCTIONS ET PREDIRE :

def main():
	file = 'Automobile.data.csv'
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
		testSet2.append(testSet[i][1:])
		valeurAttendue.append([testSet[i][0]])
	# test model
	predictionsPK = getPredictionsImpreciseKernel(dataset, columnWithClassResponse=0, testSet=testSet2, margeEpsilon=0)
	# La colonne avec la réponse de classe doit être 0 ou -1 (1ere ou dernière colonne du dataset passé en parametre)
	predictionsIK = getPredictionsImpreciseKernel(dataset, columnWithClassResponse=0, testSet=testSet2, margeEpsilon=0.1)
	print('predictions PK =', predictionsPK)
	print('valeur atendue :', valeurAttendue)
	print('predictions IK =', predictionsIK)
	accuracyPK = getAccuracyImpreciseKernel(testSet, predictionsPK, (0))
	accuracyIK = getAccuracyImpreciseKernel(testSet, predictionsIK, (0))
	print('Accuracy Precise Kernel : ', accuracyPK)
	print('Accuracy Imprecise Kernel : ', accuracyIK)
	return accuracyPK, accuracyIK

#main()


# 14 ) Fonction qui lance plusieurs fois le test avec des repartitions de donnee differentes
# Permet d'obtenir des moyennes
# Attention, ce n'est pas de la cross validation pour autant.
def launchXTimes(times):
	result = []
	meanPK = 0
	meanIK = 0
	for i in range(times):
		# random.seed(i)
		print('\n \n Resultats de l\'iteration : ', i + 1)
		result.append(main())
		meanPK += result[i][0]
		meanIK += result[i][1]
	meanPK /= times
	meanIK /= times
	print('\n \n Resultats precis moyens : ', meanPK, '\n \n Resultats imprecis moyen : ',meanIK)



launchXTimes(10)


# Tableaau des identifiants des réponses IRIS :
# 0 = setosa
# 1 = versicolor
# 2 = virginica
# Colonnes :
#SE_L, N, 4, 0
#SE_W, N, 4, 0
#PE_L, N, 4, 0
#PE_W, N, 4, 0
#ESPECE, C, 20



# Tableau colonnes diabetes :
# Pregnancies
# Glucose
# BloodPressure
# SkinThickness
# Insulin
# BMI
# DiabetesPedigreeFunction
# Age
# Outcome
