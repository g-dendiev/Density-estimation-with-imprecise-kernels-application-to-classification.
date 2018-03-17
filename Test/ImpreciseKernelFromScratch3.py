# NAIVE BAYE SFROM SCRATCH AVEC : https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/

# Implementation de la maximalite pour comparer les classes entre elles

# 0) Import des outils utiles des implementations du kernel triangulaire.

from classes.Kernels.TriangularKernel import TriangularKernel
from classes.KernelContext import KernelContext
import matplotlib.pyplot as plt
import time
import math

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
				# Passage quantitatif a qualitatif
				# -> utilisation a éviter, notre publi se base sur des datasets quantitatifs
				if dataset[i][j] not in dictionnaire.keys():
					dictionnaire[dataset[i][j]] = dictKey
					dictKey += 1
					dataset[i][j] = dictionnaire[dataset[i][j]]
				else:
					dataset[i][j] = dictionnaire[dataset[i][j]]
	return dataset
#Test import :
#filename = 'BreastTissue_nettoye.data.csv'
#dataset = loadCsv(filename=filename)
#print('load data : ',filename, 'avec', len(dataset) ,'lignes ')



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

# Test split :
#dataset = [[1],[2],[3],[4],[5]]
#splitRatio = (2/3)
#train, test = splitDataset(dataset=dataset, splitRatio=splitRatio)
#print('on a split avec en train :',train,'et en test : ',test)



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

# Test separation des donnees :
#dataset = [[1,20,1],[2,34,4],[1,34,4],[2,20,1]]
#separated0 = separateByClass(dataset=dataset, columnWithClassResponse=0)
#separated1 = separateByClass(dataset=dataset, columnWithClassResponse=1)
#separated_1 = separateByClass(dataset=dataset, columnWithClassResponse=-1)
#print('separation 1 sur col 0', separated0)
#print('separation 2 sur col 1 : ', separated1) #print error !
#print('separation 3 sur col -1', separated_1)




# 4) Initialisation du hOpt pour un dataset donnee
# Note : on passe en parametre toutes les donnees d'une var.
# On fera donc n appels a cette fonction pour n var differentes.
# Le but etant d'avoir un hOpt avec toutes les donnees avant de faire la regression classe par classe.

from statistics import stdev
from statistics import mean

def InitHOptKernelImprecise(dataset):
	if len(dataset) == 0:
		print('Erreur le dataset est vide pour l\'initialisation de hOpt')
		return 0
	sigma = stdev(dataset)
	mean2 = mean(dataset)
	if sigma != 0 :
		hOpt = 1.06 * sigma  * (len(dataset)) ** (-1 / 5)
	else :
		#print('ELSE     ****************************************')
		hOpt = 1.06 * mean2 * (len(dataset)) ** (-1 / 5)
	#print('\ndataset = ',dataset,'\nhOpt =',hOpt,'\n')
	return hOpt

#Test InitHOptKernelImprecise:
#dataset = [27,32,34,25]
#hOpt = InitHOptKernelImprecise(dataset)
#print(' hOpt = ',hOpt)



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

# Test separation des donnees sans colonne reponse :
#dataset = [[1,20,1],[2,34,4],[1,34,4],[2,20,1]]
#separated0 = separateByClassWithoutResponse(dataset=dataset, columnWithClassResponse=0)
#separated1 = separateByClassWithoutResponse(dataset=dataset, columnWithClassResponse=1)
#separated_1 = separateByClassWithoutResponse(dataset=dataset, columnWithClassResponse=-1)
#print('separation 1 sur col 0', separated0)
#print('separation 2 sur col 1 : ', separated1) #print error !
#print('separation 3 sur col -1', separated_1)



# 6) On enleve la frequence presente en derniere colonne des calculs issus de nos regressions

def separateFrequence(generalLowProbabilities,generalHightProbabilities, generalPreciseProbabilities):
	#print('\n separate frequence : \ngeneralLowProbabilities : ',generalLowProbabilities,'\n generalHightProbabilities : ',generalHightProbabilities, '\ngeneralPreciseProbabilities : ',generalPreciseProbabilities)
	frequence_y = {}
	generalLowProbabilitiesWithoutFrequence = {}
	generalHightProbabilitiesWithoutFrequence = {}
	generalPreciseProbabilitiesWithoutFrequence = {}
	for classValue, tabResult in generalLowProbabilities.items():
		n = len(tabResult)
		frequence_y[classValue] = tabResult[-1] #frequence stockée en derniere position
		generalLowProbabilitiesWithoutFrequence[classValue] = []
		for i in (range(n-1)): # On recopie tout sauf la deniere colonne
			generalLowProbabilitiesWithoutFrequence[classValue].append(tabResult[i])
	for classValue, tabResult in generalHightProbabilities.items():
		n = len(tabResult)
		generalHightProbabilitiesWithoutFrequence[classValue] = []
		for i in (range(n-1)): # On recopie tout sauf la deniere colonne
			generalHightProbabilitiesWithoutFrequence[classValue].append(tabResult[i])
	for classValue, tabResult in generalPreciseProbabilities.items():
		n = len(tabResult)
		generalPreciseProbabilitiesWithoutFrequence[classValue] = []
		for i in (range(n-1)): # On recopie tout sauf la deniere colonne
			generalPreciseProbabilitiesWithoutFrequence[classValue].append(tabResult[i])
	#print('\n len generalLowProbabilitiesWithoutFrequence classe 0 : ',generalLowProbabilitiesWithoutFrequence,'\n len generalHightProbabilitiesWithoutFrequence classe 0 : ',generalHightProbabilitiesWithoutFrequence, '\n len generalPreciseProbabilitiesWithoutFrequence de classe 0 : ',generalPreciseProbabilitiesWithoutFrequence,'\n fin separate frequence.')
	return generalLowProbabilitiesWithoutFrequence,generalHightProbabilitiesWithoutFrequence,generalPreciseProbabilitiesWithoutFrequence, frequence_y

# Test separation de la frequence :
#generalLowProbabilities={1:[[.05,.4,.4,.05],[.2,.3,.3,.1],[.2,.3,.3,.1],[0.7]], 2:[[.1,.1,.1,.6],[.1,.1,.1,.6],[.1,.1,.1,.6],[0.3]]}
#generalHightProbabilities={1:[[.05,.45,.45,.05],[.2,.35,.35,.1],[.2,.35,.35,.1],[0.7]], 2:[[.1,.1,.15,.65],[.1,.1,.15,.65],[.1,.1,.15,.65],[0.3]]}
#generalPreciseProbabilities={1:[[.05,.45,.45,.05],[.2,.35,.35,.1],[.2,.35,.35,.1],[0.7]], 2:[[.1,.1,.15,.65],[.1,.1,.15,.65],[.1,.1,.15,.65],[0.3]]}
#generalLowProbabilitiesWithoutFrequence,generalHightProbabilitiesWithoutFrequence,generalPreciseProbabilitiesWithoutFrequence, frequence_y = separateFrequence(generalLowProbabilities,generalHightProbabilities, generalPreciseProbabilities)
#print('\n  generalLowProbabilitiesWithoutFrequence : ',generalLowProbabilitiesWithoutFrequence,'\n  generalHightProbabilitiesWithoutFrequence : ',generalHightProbabilitiesWithoutFrequence, '\n  generalPreciseProbabilitiesWithoutFrequence  : ',generalPreciseProbabilitiesWithoutFrequence,'\n frequence : ',frequence_y,'\n fin separate frequence.')



# 7) Calcul des probas conditionnelles jointes (hautes et basses) pour appliquer la maximalite ensuite
#  Ajout d'un parametre de penalite afin de penaliser notre produit de probas si une des probas rencontrees est nulle.
def findProbabilityImpreciseKernel(generalLowProbabilities, generalHightProbabilities, generalPreciseProbabilities,tKernelDomain, inputVector,penalite = 0.1):

	if (len(generalLowProbabilities) == 0 or len(generalHightProbabilities) == 0 or len(tKernelDomain) == 0 or len(generalPreciseProbabilities) == 0 or len(inputVector) == 0):
		print('Erreur dans un des parametre du findProbabilityImpreciseKernel')
		if (len(generalLowProbabilities) == 0):
			print('le parametre -generalLowProbabilities- est vide')
		if (len(generalHightProbabilities) == 0):
			print('le parametre -generalHightProbabilities- est vide')
		if (len(generalPreciseProbabilities) == 0):
			print('le parametre -generalPreciseProbabilities- est vide')
		if (len(tKernelDomain) == 0):
			print('le domaine -tKernelDomain- est vide')
		if(len(inputVector) == 0):
			print('le vecteur d\'entree -inputVector- est vide')

	# Creation des dictionnaires
	## stockage des resultats classe par classe du vecteur x passe en parametre
	localLowProbabilities = {}
	localHightProbabilities = {}
	localPreciseProbabilities = {} # Ici on aura un dictionnaire avec le resultat precis de chaque var de chaque classe pour notre point d'etude
	## copie des probas generale sans la frequence, et dict pour la frequence par classe
	generalLowProbabilitiesWithoutFrequence = {}
	generalHightProbabilitiesWithoutFrequence = {}
	generalPreciseProbabilitiesWithoutFrequence = {}
	frequence_y = {}

	# Copie du domaine
	tKernelDomainLow = {}
	tKernelDomainHight = {}
	tKernelDomainPrecise = {}
	tKernelDomainLow = tKernelDomain
	tKernelDomainHight = tKernelDomain
	tKernelDomainPrecise = tKernelDomain

	# Separation des calculs lies a la regression et de la frequence
	generalLowProbabilitiesWithoutFrequence,generalHightProbabilitiesWithoutFrequence,generalPreciseProbabilitiesWithoutFrequence, frequence_y = separateFrequence(generalLowProbabilities,generalHightProbabilities,generalPreciseProbabilities)
	# Partie probas precises :
	# Boucle sur les classes du dictionnaire et leur regression pour chaque var que l'on etudie
	# but : pouvoir s'en servir quand on doit penaliser notre classifieur imprecis
	for classValue, tabResult in generalPreciseProbabilitiesWithoutFrequence.items():
		#print('\n precise classValue ',classValue)
		#print('\n len tab result i = ',len(tabResult))
		# Initialisation pour la multiplication
		localPreciseProbabilities[classValue] = []

		# Attention on a un tableau de resultats a plusieurs dimensions a gerer
		# 1ere dimension = la var utilisee
		# 2e = la valeur de la regression au point d'application du domaine de regression

		# Iteration sur les differents attributs / var d'entree avec des resultats specifiques a chaque classe
		for i in range(len(tKernelDomainPrecise[classValue])):
			#print('len tKernelDomainPrecise[',classValue,']) = ',len(tKernelDomainPrecise[classValue]))
			# Initialisation de l'indice ou notre x est le plus proche d'un point ou l'on a fait la regression
			jMin = 0

			# definition d'un domaine en lien avec la var en cours
			tKernelDomainBis = []
			tKernelDomainBis.append(tKernelDomainPrecise[classValue][i])
			#print('len tKernelDomainBis precise = ',len(tKernelDomainBis[0]),'class value : ',classValue)

			# On stocke les resultats de la regression de la classe en cours sur la var en cours
			tableauVarClassEnCours = []
			tableauVarClassEnCours.append(tabResult[i])

			# On itere sur une var d'un dictionnaire contenant un tableau a 2 dimensions (cf avant)
			# But : trouver le point pour lequel on est le plus proche du point etudie (x[i]) du vecteur x en entree
			for j in range(len(tKernelDomainPrecise[classValue][i])):
				xMinTest = abs(tKernelDomainBis[0][j] - inputVector[i])
				if (abs(tKernelDomainBis[0][jMin] - inputVector[i]) > xMinTest):
					jMin = j

			# On stocke nos probas de classe precises en cours
			if tableauVarClassEnCours[0][jMin] == 0:
				localPreciseProbabilities[classValue].append(0.01)
			else :
				localPreciseProbabilities[classValue].append(tableauVarClassEnCours[0][jMin])
	#print('\n\nlocalPreciseProbabilities = ',localPreciseProbabilities)


	# Partie probas basses :
	# Boucle sur les classes du dictionnaire et leur regression pour chaque var que l'on etudie
	for classValue, tabResult in generalLowProbabilitiesWithoutFrequence.items():
		#print('\n low classValue ', classValue)
		#print('\n len tab result i = ', len(tabResult))
		# Initialisation pour la multiplication
		localLowProbabilities[classValue] = 1

		# Attention on a un tableau de resultats a plusieurs dimensions a gerer
		# 1ere dimension = la var utilisee
		# 2e = la valeur de la regression au point d'application du domaine de regression

		# Iteration sur les differents attributs / var d'entree avec des resultats specifiques a chaque classe
		for i in range(len(tKernelDomainLow[classValue])):
			#print('len tKernelDomainLow[',classValue,']) = ',len(tKernelDomainLow[classValue]))
			# Initialisation de l'indice ou notre x est le plus proche d'un point ou l'on a fait la regression
			jMin = 0

			#print(' classe : ',classValue, 'valeur dans precise element : ',i,' = ',localPreciseProbabilities[classValue][i])

			# definition d'un domaine en lien avec la var en cours
			tKernelDomainBis = []
			tKernelDomainBis.append(tKernelDomainLow[classValue][i])

			# On stocke les resultats de la regression de la classe en cours sur la var en cours
			tableauVarClassEnCours = []
			tableauVarClassEnCours.append(tabResult[i])
			#print('len tKernelDomainBis low = ',len(tKernelDomainBis[0]),'class value : ',classValue)


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
				# On penalise par un petit coef fois la proba precise. Si ona 0 en precis alors on exclu la classe ensuite dans le predict !!
				#print('classe : ',classValue,' valeur nulle en probabilite basse au point d\'etude ',inputVector[i],', du vecteur : ',inputVector,', on penalise par : ',penalite)
				localLowProbabilities[classValue] *= penalite*localPreciseProbabilities[classValue][i]
			else:
				# On multiplie nos probas de classe par la probas de classe de la var en cours
				localLowProbabilities[classValue] *= tableauVarClassEnCours[0][jMin]

		# multiplication par la frequence et non par le tableau contenant la frequence pour la classe en cours
		localLowProbabilities[classValue] *= frequence_y[classValue][0]


	# Partie probas hautes :
	# Boucle sur les classes du dictionnaire et leur regression pour chaque var que l'on etudie
	for classValue, tabResult in generalHightProbabilitiesWithoutFrequence.items():
		#print('\n precise classValue ', classValue)
		#print('\n len tab result i = ', len(tabResult))
		# Initialisation pour la multiplication
		localHightProbabilities[classValue] = 1

		# Attention on a un tableau de resultats a plusieurs dimensions a gerer
		# 1ere dimension = la var utilisee
		# 2e = la valeur de la regression au point d'application du domaine de regression

		# Iteration sur les differents attributs / var d'entree avec des resultats specifiques a chaque classe
		for i in range(len(tKernelDomainHight[classValue])):
			#print('len tKernelDomainHight[',classValue,']) = ',len(tKernelDomainHight[classValue]))
			# Initialisation de l'indice ou notre x est le plus proche d'un point ou l'on a fait la regression
			# Remarque : on a le meme domaine que pour les probas basses
			# Donc pour un meme vecteur x en entree on aura en basse et en haute les memes points d'application de la
			# regression. Ce qui permet d'avoir les probas jointes hautes et basses conditionnellement aux classes.
			jMin = 0

			#print(' classe : ',classValue, 'valeur dans precise element : ',i,' = ',localPreciseProbabilities[classValue][i])

			# definition d'un domaine en lien avec la var en cours
			tKernelDomainBis = []
			tKernelDomainBis.append(tKernelDomainHight[classValue][i])
			#print('len tKernelDomainBis hight = ',len(tKernelDomainBis[0]),'class value : ',classValue)


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
				localHightProbabilities[classValue] *= penalite*localPreciseProbabilities[classValue][i]
			else:
				# On multiplie nos probas de classe par la probas de classe de la var en cours
				localHightProbabilities[classValue] *= tableauVarClassEnCours[0][jMin]
		# multiplication par la frequence et non par le tableau contenant la frequence pour la classe en cours
		localHightProbabilities[classValue] *= frequence_y[classValue][0]
	#print(' local low : ',localLowProbabilities,'\nlocal Hight : ',localHightProbabilities,'\n\n')
	return localLowProbabilities, localHightProbabilities

# Test separation de la frequence :
#generalLowProbabilities={1:[[.05,.4,.4,.05],[.2,.3,.3,.1],[.2,.3,.3,.1],[0.7]], 2:[[.1,.1,.1,.6],[.1,.1,.1,.6],[.1,.1,.1,.6],[0.3]]}
#generalHightProbabilities={1:[[.05,.45,.45,.05],[.2,.35,.35,.1],[.2,.35,.35,.1],[0.7]], 2:[[.1,.1,.15,.65],[.1,.1,.15,.65],[.1,.1,.15,.65],[0.3]]}
#generalPreciseProbabilities={1:[[.05,.45,.45,.05],[.2,.35,.35,.1],[.2,.35,.35,.1],[0.7]], 2:[[.1,.1,.15,.65],[.1,.1,.15,.65],[.1,.1,.15,.65],[0.3]]}
#tKernelDomain={1:[[0.85,0.95,1.05,1.15],[1.85,1.95,2.05,2.15],[3.85,3.95,4.05,4.15]], 2:[[3.85,3.95,4.05,4.15],[5.85,5.95,6.05,6.15],[7.85,7.95,8.05,8.15]]}
#inputVector=[0.9,1.9,3.9]
#localLowProbabilities, localHightProbabilities = findProbabilityImpreciseKernel(generalLowProbabilities, generalHightProbabilities, generalPreciseProbabilities,tKernelDomain, inputVector,penalite = 0.1)
#print('localLowProbabilities :',localLowProbabilities, 'local hight :',localHightProbabilities)



# 8) Calcul des pobabilites pour une classe donnee en parametre
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
	#print('max domain =',maxDomain,'min domain =',minDomain)
	if (math.floor(maxDomain-minDomain) != 0):
		stepLinspace = (math.floor(maxDomain-minDomain))/nbPointsRegressionMoins1
	else:
		stepLinspace = (maxDomain - minDomain) / nbPointsRegressionMoins1
	#print('step : ', stepLinspace)
	# Creation des tableaux contenant les resultats pour la var en cours et la classe en cours
	lowProbabilities = []
	hightProbabilities = []
	preciseProbabilities = []
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
		structHWithoutEpsilon = {'potentialHValue': -1, 'maxedValue': -1}

		# Calculs de f(hMax), et f(hMin)
		structHMax = tKernelTriClass.computeHMaxFromInterval(pt, h, epsilon)
		structHMin = tKernelTriClass.computeHMinFromInterval(pt, h, epsilon)
		structHWithoutEpsilon = tKernelTriClass.computeHMaxFromInterval(pt, h, 0)

		# Stockage successifs des resultats
		hightProbabilities.append(structHMax['maxedValue'])
		lowProbabilities.append(structHMin['minValue'])
		preciseProbabilities.append(structHWithoutEpsilon['maxedValue'])

	#print('\n\n\nlowProbabilities : ',lowProbabilities, '\nhightProbabilities : ',hightProbabilities, '\npreciseProbabilities : ',preciseProbabilities, '\ngeneralDomain : ',generalDomain)
	# On retourne egalement le domaine pour pouvoir ensuite faire les probas jointes hautes et basses.
	return lowProbabilities, hightProbabilities, preciseProbabilities, generalDomain


# 9 ) Fonction qui va etre appelee une fois pour calculer les probas imprecis de chaque var pour chaque classe.
# La marge est celle passée dans la fonction principale en parametre, elles est en pourcentage.
def calculateClassProbabilitiesImpreciseKernel(dataset, columnWithClassResponse, margeEpsilon):
	# Declaration / initialisation des variables
	N = len(dataset)
	dataset2 = []
	hOpt = []
	lowProbabilities = {}
	hightProbabilities = {}
	preciseProbabilities = {}
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
		preciseProbabilities[classValue] = []
		generalDomain[classValue] = []

		# Initialisation de la frequence de la classe : estimateur de p(Y)
		frequence_y = len(classDataset) / N

		# Si on a au moins une var d'entree, alors on lance la machine. Sinon message d'erreur.
		if(len(classDatasetWithColSeparated) != 0):
			for i in range(len(classDatasetWithColSeparated)):
				# Calcul toute la densité sur la classe et ensuite on assigne les prbas selon les points en entrée et en sortie
				generalLowProbabilities, generalHightProbabilities, generalPreciseProbabilities, generalDomainI = calculateProbabilityImpreciseKernel(datasetTotalVar=dataset2Separated[i],datasetClass=classDatasetWithColSeparated[i], h=hOpt[i], epsilon=margeEpsilon * hOpt[i])

				# On crée un tableau avec chaque classe et les données pour chaque var !
				# On fait donc le produit de toutes les probas basses dans la fonction predict !
				lowProbabilities[classValue].append(generalLowProbabilities)
				hightProbabilities[classValue].append(generalHightProbabilities)
				preciseProbabilities[classValue].append(generalPreciseProbabilities)
				generalDomain[classValue].append(generalDomainI)

			# On stocke en dernier sous tableau la frequence d'apparition de la classe !
			lowProbabilities[classValue].append([frequence_y])
			hightProbabilities[classValue].append([frequence_y])
			preciseProbabilities[classValue].append([frequence_y])
		else :
			print('ERREUR : IL N\'EXISTE AUCUNE VARIABLE POUR PREDIRE !!!!!!')
			break
	return lowProbabilities, hightProbabilities, preciseProbabilities, generalDomain


# 10) On definit la fonction de prediction de classe avec la maximalite comme critere de comparaison
def predictImpreciseKernel(lowProbabilities, hightProbabilities, preciseProbabilities, generalDomain, inputVector, margeEpsilon):
	# Declaration des variables
	bestLabel = []
	classLabel = []
	dominatedClass = []
	localLowProbabilities = {}
	localHightProbabilities = {}

	# Initialisation
	localLowProbabilities, localHightProbabilities = findProbabilityImpreciseKernel(lowProbabilities, hightProbabilities, preciseProbabilities, generalDomain, inputVector)

	#print('\nmarge epsilon : ',margeEpsilon,'\nlocal low :',localLowProbabilities,'\n local hight :',localHightProbabilities)

	# On itere sur l'ensemble des classes du tableau de probas basses qui ne sont pas dominees
	for classValueLow, lowProba in localLowProbabilities.items():
		# On stocke tous les labels de classe
		classLabel.append(classValueLow)

		# On verifie que la classe en cours n'est pas dominee
		if classValueLow not in dominatedClass:

			# Boucle de comparaison avec les probas hautes des autres classes non dominees
			for classValueHight, hightProba in localHightProbabilities.items():
				if hightProba == 0 :
					dominatedClass.append(classValueHight)
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

	if len(bestLabel) == 0 :
		# Si l'ensemble est vide, on retourne toutes les classes -> on est imprecis a mort !
		return classLabel
	else :
		# Sinon on retourne le sous ensemble des labels possibles.
		return bestLabel


# 11) Predictions sur un jeu de test complet :
def getPredictionsImpreciseKernel(dataset, columnWithClassResponse, testSet, margeEpsilon):
	# Declaration
	predictions = []

	# Initialisation des probas hautes et basses, ainsi que du domaine de chaque variable
	lowProbabilities, hightProbabilities, preciseProbabilities, generalDomain = calculateClassProbabilitiesImpreciseKernel(dataset,columnWithClassResponse,margeEpsilon)

	#print('marge epsilon = ',margeEpsilon,'\nget prediction imprecise, low = ',lowProbabilities,'\nhight = ',hightProbabilities,'\nprecise = ',preciseProbabilities)

	# Calcul de prediction pour chaque donnee de test
	for i in range(len(testSet)):
		result = predictImpreciseKernel(lowProbabilities, hightProbabilities, preciseProbabilities, generalDomain, testSet[i], margeEpsilon)
		predictions.append(result)

	# On retourne le taleau complet des predictions
	return predictions


# 12 ) Moyenne des erreurs :
def getAccuracyImpreciseKernel(testSet, predictions, columnWithClassResponse,u=0.65):
	correct = 0
	for x in range(len(testSet)):
		# Penalite imprecise avec le fontionnement (alpha/card(Y) - beta/(card(Y)^2)), u -> quand card Y = 2
		if u == 0.5: # alpha=1 et beta = 0 -> a eviter !
			if testSet[x][columnWithClassResponse] in predictions[x]:
				correct += 1/len(predictions[x])
		else:
			if u == 0.65: # alpha=1.6 et beta = 0.6
				if testSet[x][columnWithClassResponse] in predictions[x]:
					correct += (1.6 / len(predictions[x]) - 0.6 / (len(predictions[x])**2))
			else:
				if u == 0.80: # alpha=2.2 et beta = 1.2
					if testSet[x][columnWithClassResponse] in predictions[x]:
						correct += (2.2 / len(predictions[x]) - 1.2 / (len(predictions[x]) ** 2))
				else:
					print('erreur dans le passage de parametre u qui doit être a valeur dans [0.5 , 0.65, 0.80]')

	return (correct / float(len(testSet))) * 100.0



# 13) Fonction qui permet de voir si les resultats imprecis données contiennent bien la bonne valeur.
# Test sur tous les jeux de donnees.
# But : avoir  les stats selon les jeux de donnees pour ensuite faire les graphes.
def convertPreciseAndImprecisePredictionsToStats(predictions,datasets):
	statsImpreciseResults = {}

	for dataset in datasets:


		#statsImpreciseResults[dataset]=[]
		goodImprecisePrediction = 0
		goodPrecisePrediction = 0
		n = len(predictions[dataset])
		# Iteration sur le nombre de resultats imprecis du dataset
		for i in range(len(predictions[dataset])):
			# On a un tableau de tableaux à 3 dimensions avec :
			# - 0 : valeurs imprecises
			# - 1 : valeur attendue
			# - 2 : valeur precise
			if predictions[dataset][i][1][0] in predictions[dataset][i][0]:
				goodImprecisePrediction += 1
			if predictions[dataset][i][1] == predictions[dataset][i][2]:
				goodPrecisePrediction += 1
		if n > 0:
			statsImpreciseResults[dataset] = [(goodImprecisePrediction*100/n),(goodPrecisePrediction*100/n)]
	return statsImpreciseResults


# 14)Code qui prend en parametre les statistiques (TP) precises et imprecises de tous les datasets pour en faire un graphe
# Le graphe depend du epsilon passe en parametre, du split dans le dataset et du u (u65, u80)

def statsToGraph(stats, splitRatio, margeEpsilon):
	date = time.localtime()
	title = 'graph with epsilon = '+str(margeEpsilon)+' and split ratio = '+str(splitRatio)
	titlePDF = '/Users/USER/Guillaume/UTC/GI05_A17/TX02/Code_TX_A16_P-Wachalski_G-Dendievel/tx_kde/Test/'+title+str(date)+'.pdf'
	plt.grid(True)
	plt.title(title)
	plt.plot([0,100],[0,100],linewidth=0.8)

	statsTreated = []
	statsCleaned = {}

	# a : breast 106 /10 /6
	# b : iris 150 /4 /3
	# c : wine 178 /13 /3
	# d : auto 205 /26 /7
	# e : seed 210 /7 /3
	# f : glass 214 / 9 / 7
	# g : forest 325 /27 /4
	# h : derma 366 /34 /6
	# i : diabete 769/8/2
	# j : segment 2310 /19 /7

	# On inverse le dictionnaire selon les points et on ajoute les cigles si on a le meme point d'application
	for stat in stats:
		if stat not in statsTreated :

			if stat[0:3] == 'Bre':
				statsCleaned[stats[stat][1], stats[stat][0]] = 'a'
			if stat[0:3] == 'iri':
				statsCleaned[stats[stat][1], stats[stat][0]] = 'b'
			if stat[0:3] == 'win':
				statsCleaned[stats[stat][1], stats[stat][0]] = 'c'
			if stat[0:3] == 'aut':
				statsCleaned[stats[stat][1], stats[stat][0]] = 'd'
			if stat[0:3] == 'see':
				statsCleaned[stats[stat][1], stats[stat][0]] = 'e'
			if stat[0:3] == 'gla':
				statsCleaned[stats[stat][1], stats[stat][0]] = 'f'
			if stat[0:3] == 'for':
				statsCleaned[stats[stat][1], stats[stat][0]] = 'g'
			if stat[0:3] == 'der':
				statsCleaned[stats[stat][1], stats[stat][0]] = 'h'
			if stat[0:3] == 'dia':
				statsCleaned[stats[stat][1], stats[stat][0]] = 'i'
			if stat[0:3] == 'seg':
				statsCleaned[stats[stat][1], stats[stat][0]] = 'j'

			#statsCleaned[stats[stat][1],stats[stat][0]] = stat[0:3]
			#print('stat clean = ',statsCleaned)
			statsTreated.append(stat)
			for otherStats in stats :
				if otherStats not in statsTreated :
					if stats[stat][1] == stats[otherStats][1] and stats[stat][0] == stats[otherStats][0] :
						statsCleaned[stats[stat][1], stats[stat][0]]+=(','+otherStats[0])
						statsTreated.append(otherStats)

	# Boucle sur le dico
	for stat in statsCleaned:
		precis = stat[0]
		imprecis = stat[1]
		#print(' valeur precise :',precis,'\n valeur imprecise :',imprecis)
		plt.plot(precis, imprecis,"b",marker="+")#"b", linewidth=0.8, marker="*", label="Trajet")
		# annotatin : -1 en x et +2 en y
		annoteX = precis - 1
		annoteY = imprecis + 2
		annoteValue = statsCleaned[stat]
		plt.annotate(annoteValue, xy=(precis, imprecis), xytext=(annoteX, annoteY))

	# Fin boucle su dico
	plt.axis([-5, 110, 0, 110])
	plt.xlabel('Precise accuracy')
	plt.ylabel('Imprecise accuracy')
	#plt.legend()
	plt.savefig(titlePDF)
	plt.clf()
	return

#statsToGraph({'datatest' : [90,30], 'etetetet' : [80,40],'prttetet' : [80,40]},0.5,0.4)



# 15) code pour lancer les fonctions une fois et predire :

def launch(file,splitRatio,rand,columnWithClassResponse=0,margeEpsilon=0.2):
	random.seed(rand)
	dataset = loadCsv(file)
	trainingSet, testSet = splitDataset(dataset, splitRatio)
	# prepare model
	#print('Split ', len(dataset), ' rows into train=', len(trainingSet), ' and test=', len(testSet), ' rows')
	# summaries = summarizedByClass(trainingSet,columnWithClassResponse=4)
	# print('testSet = ',testSet)
	testSet2 = []
	valeurAttendue = []
	if columnWithClassResponse == 0:
		for i in range(len(testSet)):
			testSet2.append(testSet[i][1:])
			valeurAttendue.append([testSet[i][0]])
	else :
		for i in range(len(testSet)):
			testSet2.append(testSet[i][0:-1])
			valeurAttendue.append([testSet[i][-1]])
	# test model
	predictionsPK = getPredictionsImpreciseKernel(dataset, columnWithClassResponse=columnWithClassResponse, testSet=testSet2, margeEpsilon=0)
	# La colonne avec la réponse de classe doit être 0 ou -1 (1ere ou dernière colonne du dataset passé en parametre)
	predictionsIK = getPredictionsImpreciseKernel(dataset, columnWithClassResponse=columnWithClassResponse, testSet=testSet2, margeEpsilon=margeEpsilon)
	#print('predictions PK =', predictionsPK)
	#print('valeur atendue :', valeurAttendue)
	#print('predictions IK =', predictionsIK)
	accuracyPK = getAccuracyImpreciseKernel(testSet, predictionsPK, (columnWithClassResponse))
	accuracyIK65 = getAccuracyImpreciseKernel(testSet, predictionsIK, (columnWithClassResponse))
	#print('Accuracy Precise Kernel : ', accuracyPK)
	#print('Accuracy Imprecise Kernel u65 : ', accuracyIK65)
	accuracyIK80 = getAccuracyImpreciseKernel(testSet, predictionsIK, (columnWithClassResponse),0.80)
	#print('Accuracy Imprecise Kernel u80 : ', accuracyIK80)

	###########
	# Retourner toutes les predictions precises et imprecises
	# pour que l'on puisse faire les stats avec FP, FN, TP, TN
	###########

	return accuracyPK, accuracyIK65, accuracyIK80, predictionsPK, valeurAttendue, predictionsIK

# 16 ) Fonction qui lance plusieurs fois le test avec des repartitions de donnee differentes
# Permet d'obtenir des moyennes
# Attention, ce n'est pas de la cross validation pour autant.
def launchXTimes(times,margeEpsilon,splitRatio,datasets):
	file = open("/Users/USER/Guillaume/UTC/GI05_A17/TX02/Code_TX_A16_P-Wachalski_G-Dendievel/tx_kde/Test/results.txt","a")
	date = time.localtime()
	file.write("\n\n\n***********************\n Date : "+str(date)+"\n\nEpsilon : "+str(margeEpsilon)+"\nSplit ratio :"+str(splitRatio)+"\n\n")
	print(' Epsilon = ',margeEpsilon)
	print(' Split ratio = ', splitRatio)
	# Creation d'un dictionnaire qui va contenir les resultats imprecis (de taille superieur a 1),
	# et les resultats precis et attendus correspondants
	impreciseResults = {}
	for dataset in datasets :
		lenMax = 0
		if dataset[0:3] == 'Bre' or dataset[0:3] == 'der':
			lenMax = 6
		if dataset[0:3] == 'iri' or dataset[0:3] == 'win' or dataset[0:3] == 'see':
			lenMax = 3
		if dataset[0:3] == 'aut' or dataset[0:3] == 'gla' or dataset[0:3] == 'seg':
			lenMax = 7
		if dataset[0:3] == 'for':
			lenMax = 4
		if dataset[0:3] == 'dia':
			lenMax = 2
		impreciseResults[dataset]=[]
		result = []
		meanPK = 0
		meanIK65 = 0
		meanIK80 = 0
		ratioImprecis = 0
		nbResultPartiel = 0
		imprecisTotal = 0
		imprecisPartiel = 0
		denominateurImprecis = 0
		imprecisPartielCarre = 0
		stdevLongueurResultsImprecis = 0
		if dataset in ['forestType.data.csv','automobile.data.csv','wine.data.csv','BreastTissue_nettoye.data.csv','letter-recognition.data.csv'] :
			columnWithClassResponse = 0
		else :
			columnWithClassResponse = -1

		for i in range(times):
			random.seed(i)
			#print('\n \n Resultats de l\'iteration : ', i + 1,'\n Dataset : ',dataset)
			result.append(launch(dataset,splitRatio,i,columnWithClassResponse,margeEpsilon))
			meanPK += result[i][0]
			meanIK65 += result[i][1]
			meanIK80 += result[i][2]
			predictionsPK = result[i][3]
			valeurAttendue = result[i][4]
			predictionsIK = result[i][5]
			#print('prediction ik : ',predictionsIK,'\nvalue : ',valeurAttendue,'\nprediction pk : ',predictionsPK)

			# Boucle sur les resultats imprecis, on stocke imprecis, valeur attendue et precise si len(imprecis) > 1
			for j in range(len(predictionsIK)):
				if len(predictionsIK[j]) > 1:
					impreciseResults[dataset].append([predictionsIK[j],valeurAttendue[j],predictionsPK[j]])
					#print('Results imprecis = ',impreciseResults)
					if len(predictionsIK[j]) == lenMax :
						imprecisTotal += 1
					else :
						imprecisPartiel += len(predictionsIK[j])
						imprecisPartielCarre += (len(predictionsIK[j])*len(predictionsIK[j]))
						nbResultPartiel += 1
			denominateurImprecis += (len(predictionsIK))
		#print('\n nb imprecis partiel :',nbResultPartiel,'\n nb imprecis total : ',imprecisTotal,'\n denominateur imprecis :',denominateurImprecis)
		ratioImprecis = ((len(impreciseResults[dataset]))/denominateurImprecis)*100
		ratioImprecisPartiel = (nbResultPartiel/denominateurImprecis)*100
		ratioImprecisTotal = (imprecisTotal/denominateurImprecis)*100
		if nbResultPartiel != 0 :
			moyenneLongueurResultsImprecis = imprecisPartiel/nbResultPartiel
			stdevLongueurResultsImprecis = ((1/nbResultPartiel)*(imprecisPartielCarre) - moyenneLongueurResultsImprecis*moyenneLongueurResultsImprecis)**(1/2)
		else :
			moyenneLongueurResultsImprecis = 0
			stdevLongueurResultsImprecis = 0




####################
		# Boucle sur les predictions et la valeur attendue.
		# But : obtenir des vecteur ayant tous les cas a plusieurs classes du vecteur imprecis
		# Ensuite on aura le vecteur de toutes les valeurs attendues précises
		# Et de toutes les predictions precises
		# Stocker tout ça dans des variables en fonction du dataset !
		####################

		meanPK /= times
		meanIK65 /= times
		meanIK80 /= times
		print('\n Dataset : ',dataset,'\n Resultats precis moyens : ', meanPK, '\n Resultats imprecis moyen u65 : ',meanIK65,'\n Resultats imprecis moyen u80 : ',meanIK80,'\n Ratio imprecis : ',ratioImprecis,'\n Ratio imprecis total : ',ratioImprecisTotal,'\n Ratio imprecis partiel : ',ratioImprecisPartiel,'\n Moyenne du nombre de classes retournées en imprécis partiel : ',moyenneLongueurResultsImprecis,'\n ecart type nb classes imprecises partiel : ',stdevLongueurResultsImprecis,'\nResultats imprecis [[[IK_1],TrueValue_1,PK_1],[[IK_2],TrueValue_2,PK_2],...]: ',impreciseResults[dataset],'\n')
		file.write("\n Dataset : "+str(dataset)+"\n Resultats precis moyens : "+str(meanPK)+ "\n Resultats imprecis moyen u65 : "+str(meanIK65)+"\n Resultats imprecis moyen u80 : "+str(meanIK80)+"\n Ratio imprecis : "+str(ratioImprecis)+"\n Ratio imprecis total : "+str(ratioImprecisTotal)+"\n Ratio imprecis partiel : "+str(ratioImprecisPartiel)+"\n Moyenne du nombre de classes retournées en imprécis partiel : "+str(moyenneLongueurResultsImprecis)+"\n ecart type nb classes imprecises partiel : "+str(stdevLongueurResultsImprecis)+"\nResultats imprecis [[[IK_1],TrueValue_1,PK_1],[[IK_2],TrueValue_2,PK_2],...]: "+str(impreciseResults[dataset])+"\n")

	statsImpreciseResults = convertPreciseAndImprecisePredictionsToStats(impreciseResults, datasets)
	#print('Stats : ',statsImpreciseResults)
	statsToGraph(statsImpreciseResults, splitRatio, margeEpsilon)
	file.write("\n\n\n***********************")
	file.close()
	return


def main():
	for margeEpsilon in [0.1,0.2,0.4]:
		for splitRatio in [0.3,0.5,0.75]: # Calculer le ratio d'imprécis pour pouvoir mettre en lien avec notre tableau !
			launchXTimes(10,margeEpsilon,splitRatio,['iris.data.csv','wine.data.csv','diabetes.data.csv'])# 'BreastTissue_nettoye.data.csv','automobile.data.csv','seeds_dataset.data.csv','glass_clean.data.csv','forestType.data.csv','dermatology_dataset.data.csv','segment.data.csv'

main()



# Organisation : nb valeur /nb var /nb class

# a : breast 106 /10 /6
# b : iris 150 /4 /3
# c : wine 178 /13 /3
# d : auto 205 /26 /7
# e : seed 210 /7 /3
# f : glass 214 / 9 / 7
# g : forest 325 /27 /4
# h : derma 366 /34 /6
# i : diabete 769/8/2
# j : segment 2310 /19 /7

# Voir pour faire tourner random forest sur les jeux de donnees afin d'avoir une idee des resultats.
