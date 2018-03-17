# Aim

The goal of this code is to realise an imprecise kernel based on the Triangular Kernel. This was in two 6-month's projects made à the "Universitée de Technologie de Compiègne", in France. You can find the two subjects in french at the end of the README.

# Article (link comming)

This code is the test part of a publication, named "Dendity estimation with imprecise kernels : application to classification".

# Run

If you want to test on your own, you can follow those steps :

  - Donwload all the all project
  - Go to the Test repository and launch the ImpreciseKernelFromScratch3.py
        -> Note : I used the PyCharm IDE to do the project on mac OS.
                  You may change some folders to the datasets to make it work.

Note : This manipulation will lead to use all pairs of parameters made with epsilon in [10%, 20%, 40%] and a separation of your datas in [ 30%, 50%, 75%]. The bases datasets are : diabetes.data.csv, iris.data.csv and wine.data.csv. So that it won't take too much time to compute it all.


# Results :

  - You will get somes results displayed in your IDE console
  - Some graphics will be saved with the date and the some parameters (epsilon, splitRatio,...), in the working directory.
        -> names will be like : 
            "graph with epsilon = 0.4 and split ratio = 0.3time.struct_time(tm_year=2018, tm_mon=2, tm_mday=27, tm_hour=6, tm_min=21, tm_sec=5, tm_wday=1, tm_yday=58, tm_isdst=0).pdf"
  - Also all statistical and imprecise classification results (in results.txt, by concatenation, so that you can keep the older results).
        -> See the results.txt in Test repository to see an example.


Note : The graphics, named like graph with epsilon = 0.4 and split ratio = 0.3time.struct_time(tm_year=2018, tm_mon=2, tm_mday=27, tm_hour=6, tm_min=21, tm_sec=5, tm_wday=1, tm_yday=58, tm_isdst=0).pdf, you have in the test repository have been made with those datasets extracts from the UCI site : https://archive.ics.uci.edu/ml/datasets.html :
 
Organisation : 
- name : nb values /nb variables /nb class
- a : breast 106 /10 /6
- b : iris 150 /4 /3
- c : wine 178 /13 /3
- d : auto 205 /26 /7
- e : seed 210 /7 /3
- f : glass 214 / 9 / 7
- g : forest 325 /27 /4
- h : derma 366 /34 /6
- i : diabete 769/8/2
- j : segment 2310 /19 /7

# Parameters

Feel free to :

  - test others datasets in '.csv', essentially numerics datasets.
  - test others epsilons or splitRatio


# Go further

To understand how this works, see the article : LINK COMMING.

There is a little definition of each fonction used in ImpreciseKernelFromScratch3.py before the fonciton is implemented. Not for all but there is sometimes some unitary tests.


In the Classes repository contains all implementations of kernels (triangular and Epanechikov). You can see it if you want to.

# Subjects :
## Autumn 2016 (6 months):

Sujet Automne 2016 :

Estimation de densité imprécise par ensemble de noyaux, application aux classifieurs naïfs

L’estimation de densité par noyaux est une méthode non-paramétrique courante pour estimer, à partir d’échantillons, des
densités de probabilités dont la forme n’est pas connue a priori. Une telle estimation peut cependant s’avérer peu fiable,
par exemple si le noyau utilisé est assez mal choisi ou si le nombre de données est peu élevés. Dans cette TX, nous
nous proposons de considérer le problème d’estimer des densités à partir d’un ensemble de noyaux, la densité estimée se
transformant alors en un ensemble possible de densités. Ces estimations imprécises pourront ensuite par exemple être
utilisées pour réaliser des prédictions prudentes à partir de classifieurs naïfs imprécis (fournissant un ensemble de
classes plutôt qu’une en cas de manque d’information). Nous attendons des candidats qu'ils:

* Prennent connaissance de la problématique d’estimation de densité (univariée) par noyaux
* Encodent (dans le langage de leur choix, avec une préférence pour python) un algorithme permettant de réaliser
  l’estimation par ensemble de noyaux
* Si le temps le permet, utilisent ces estimations dans l’extension imprécise du classifieur Bayésien Naïf

De plus, les candidats pourront:

* Comparer leurs résultats à ceux obtenus sous hypothèse Gaussienne
* S’intéresser au cas multivarié
* Essayer de traiter des jeux de données de grande taille (en nombre de labels, d'attributs et/ou d'exemples)
* Etudier des aspects plus théoriques de la problématique


## Autumn 2017 until the end of winter (8 months) :

Sujet Automne 2017 :

Sujet :	L’estimation de densité par noyaux est une méthode non-paramétrique courante pour estimer, à partir d’échantillons, des densités de probabilités dont la forme n’est pas connue a priori. Une telle estimation peut cependant s’avérer peu fiable, par exemple si le noyau utilisé est assez mal choisi ou si le nombre de données est peu élevés. Dans cette TX, nous nous proposons de prendre la suite de travaux précédents et de les appliquer à des problématiques de traitement du signal et de classification, puis de les retrasncrire dans un article scientifique. Nous attendons des candidats qu'ils: * Mettent en oeuvre les méthodes d'estimation développées sur des problématiques de classification (et éventuellement de traitement du signal) * Participent à l'écriture d'un article scientifique relatant les résultats De plus, les candidats pourront: * S’intéresser au cas multivarié * Essayer de traiter des jeux de données de grande taille (en nombre de labels, d'attributs et/ou d'exemples) Aspects des travaux attendus dans la TX (de +, “peu présent” à +++, “très présent”) : * bibliographie: ++ * formalisation: +++ * implémentation: + * tests: ++ * écriture: ++ Outils utilisés: python, LateX
