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

filename = 'pima-indians-diabetes.data.csv'
dataset = loadCsv(filename=filename)
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

dataset = [[1],[2],[3],[4],[5]]
splitRatio = (2/3)
train, test = splitDataset(dataset=dataset, splitRatio=splitRatio)
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

dataset = [[1,20,1],[2,34,4],[1,34,4],[2,20,1]]
separated0 = separateByClass(dataset=dataset, columnWithClassResponse=0)
separated1 = separateByClass(dataset=dataset, columnWithClassResponse=1)
separated_1 = separateByClass(dataset=dataset, columnWithClassResponse=-1)
#print('separation 1 sur col 0', separated0)
#print('separation 2 sur col 1', separated1)
#print('separation 3 sur col -1', separated_1)


# Calcul de données de stats :

import math
def mean(numbers):
    return sum(numbers)/len(numbers)

def stdev(numbers):
    avg = mean(numbers=numbers)
    variance = sum([pow(x-avg,2)for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

# Test : IT WORKS
numbers = [1,2,3,4,5]
#print('moyenne',mean(numbers=numbers))
#print('variance', stdev(numbers=numbers))

#Sommaire des données : mode général :
#The zip function groups the values for each attribute across our data instances
# #into their own lists so that we can compute the mean and standard deviation values for the attribute.
def summarize(dataset,columnWithClassResponse):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[columnWithClassResponse]
    return summaries

# Test summarize :
dataset = [[1,20,1],[2,34,4],[1,34,4],[2,20,1]]
summary0 = summarize(dataset=dataset, columnWithClassResponse=0)
summary1 = summarize(dataset=dataset, columnWithClassResponse=1)
summary_1 = summarize(dataset=dataset, columnWithClassResponse=-1)
print('sommaire 1 en enlevant col 0', summary0)
print('sommaire 2 en enlevant col 1', summary1)
print('sommaire 3 en enlevant col -1', summary_1)

