###########################################################################################

Pour commit proprement suivre la liste d’instruction suivante :


git pull — - rebase

git add « tous les fichiers que j’ai motif ou créé »

git commit -m « message pour l’autre à laisser »

git push

###########################################################################################

Sujet :

Estimation de densité imprécise par ensemble de noyaux, application aux classifieurs naïfs

L’estimation de densité par noyaux est une méthode non-paramétrique courante pour estimer, à partir d’échantillons, des densités de probabilités dont la forme n’est pas connue a priori. Une telle estimation peut cependant s’avérer peu fiable, par exemple si le noyau utilisé est assez mal choisi ou si le nombre de données est peu élevés. Dans cette TX, nous nous proposons de considérer le problème d’estimer des densités à partir d’un ensemble de noyaux, la densité estimée se transformant alors en un ensemble possible de densités. Ces estimations imprécises pourront ensuite par exemple être utilisées pour réaliser des prédictions prudentes à partir de classifieurs naïfs imprécis (fournissant un ensemble de classes plutôt qu’une en cas de manque d’information). Nous attendons des candidats qu'ils:

* Prennent connaissance de la problématique d’estimation de densité (univariée) par noyaux
* Encodent (dans le langage de leur choix, avec une préférence pour python) un algorithme permettant de réaliser l’estimation par ensemble de noyaux
* Si le temps le permet, utilisent ces estimations dans l’extension imprécise du classifieur Bayésien Naïf

De plus, les candidats pourront:

* Comparer leurs résultats à ceux obtenus sous hypothèse Gaussienne
* S’intéresser au cas multivarié
* Essayer de traiter des jeux de données de grande taille (en nombre de labels, d'attributs et/ou d'exemples)
* Etudier des aspects plus théoriques de la problématique

###########################################################################################

Définition/formalisation du problème auquel nous devons répondre :

Dans le cadre de l'étude du traitement du signal, nous cherchons à restreindre la fenêtre de balayage de
l'application de nos différents Kernel (triangulaire, ellispsoidal,...) afin qu'elle soit la plus petite possible
et qu'elle contienne la largeur optimale de balayage (hOpt).

Soit hOpt la largeur optimale de balayage pour une fonction f,
il faut que nous trouvions un intervalle [hMin, hMax] tel que :
    Quelque soit f : hOpt C [hMin,hMax].

Notre travail consiste donc à essayer de maximiser ce hMin et de minimiser ce hMax, afin de limiter l'étude du signal dans
l'intervalle [hMin, hMax] et ainsi gagner du temps en évitant de balayer un ensemble trop large de valeurs de h, tout en
garantissant le fait que le hOpt pour la fonction f (inconnue) appartienne à cet intervalle.
