The goal of this code is to realise an imprecise kernel based on the Triangular Kernel.

Your can see the ImpreciseKernelFromScratch3.py in Test repository to see how we did and launch it.

Classes repository contain all implementations of kernels (triangular and Epanechikov).


###########################################################################################


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


###########################################################################################

Sujet Automne 2017 :

Sujet :	L’estimation de densité par noyaux est une méthode non-paramétrique courante pour estimer, à partir d’échantillons, des densités de probabilités dont la forme n’est pas connue a priori. Une telle estimation peut cependant s’avérer peu fiable, par exemple si le noyau utilisé est assez mal choisi ou si le nombre de données est peu élevés. Dans cette TX, nous nous proposons de prendre la suite de travaux précédents et de les appliquer à des problématiques de traitement du signal et de classification, puis de les retrasncrire dans un article scientifique. Nous attendons des candidats qu'ils: * Mettent en oeuvre les méthodes d'estimation développées sur des problématiques de classification (et éventuellement de traitement du signal) * Participent à l'écriture d'un article scientifique relatant les résultats De plus, les candidats pourront: * S’intéresser au cas multivarié * Essayer de traiter des jeux de données de grande taille (en nombre de labels, d'attributs et/ou d'exemples) Aspects des travaux attendus dans la TX (de +, “peu présent” à +++, “très présent”) : * bibliographie: ++ * formalisation: +++ * implémentation: + * tests: ++ * écriture: ++ Outils utilisés: python, LateX
