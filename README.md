###########################################################################################


Pour commit proprement suivre la liste d’instruction suivante :

git pull — - rebase

git add « tous les fichiers que j’ai motif ou créé »

git commit -m « message pour l’autre à laisser »

git push

###########################################################################################

Sources à voir : https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/


###########################################################################################


Sujet :

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


Définition/formalisation du problème auquel nous devons répondre :

Dans le cadre de l'étude du traitement du signal, nous cherchons à restreindre la fenêtre de balayage de
l'application de nos différents Kernel (triangulaire, ellispsoidal,...) afin qu'elle soit la plus petite possible
et qu'elle contienne la largeur optimale de balayage (hOpt).

Soit hOpt la largeur optimale de balayage pour une fonction f,
il faut que nous trouvions un intervalle [hMin, hMax] tel que :
    Quelque soit f : hOpt C [hMin,hMax].

Notre travail consiste donc à essayer de maximiser ce hMin et de minimiser ce hMax, afin de limiter l'étude du signal dans
l'intervalle [hMin, hMax] et ainsi gagner du temps en évitant de balayer un ensemble trop large de valeurs de h, tout en
garantissant le fait que le hOpt pour la fonction f, qui est inconnue, appartienne à cet intervalle.


############################################################################################


Retour sur les Tests du Kernel Triangulaire :

- hMin : Quand on parle de hMin, on considère qu'il a la valeur de la borne inférieure de l'intervalle [hMin, hMax]
        Cet intervalle peut changer avec le temps, en fonctions des parametres rentrés et des signaux reçus.

Quand hMin tend vers 0, on obtient des résultats "en pic", et plus on s'approche de 0 plus on a de résultats nuls pour
f(hMin).
En effet on a un triangle avec une hauteur très grande afin de conserver une aire valant 1. C'est logique car note base
dépend de h et notre hauteur de 1/h donc quand h tend vers 0 on obtient des hauteurs qui tendent vers l'infini.
De ce fait :

    ° Pour donner une borne inférieure à hMin, une idée pourrait être d'étudier l'écart entre le max(f(hMin)) et le
    min(f(hMin), et ce sur un nombre de points successifs restreints (à donner en parametre). Le but serait de
    voir a quel point l'écart est grand. Si |max(f(hMin)) - min(f(hMin))| > borne_donnee_en_parametre, alors on a un h
    trop petit et on réhausse la borne inférieure de hMin d'un epsilon (donné en parametre).

    ° Il est aussi possible de donner une hauteur maximale pour notre triangle. En l'initialisant avec une valeur comme
    très grande (à définir), et en faisant en sorte que cette valeur soit modifiable par l'utilisateur. En gérant ce
    parametre, on borne inférieurement hMin. en effet hMin = 2/Borne_max_hauteur_triangle.

    ° Il faut également tester le fait que f(hMin)=0 sur un intervalle d'étude de plusieurs points (intervalle à donner
    en parametre). Si on voit que f(hMin)=0 sur un intervalle trop grand, <=> sur un ombre de points tsté trop grand,
    une fois encore on va augmenter la borne inférieure de hMin pour notre étude d'un epsilon (donné en parametre).

- hMax : Quand on parle de hMax, on considère qu'il a la valeur de la borne supérieure de l'intervalle [hMin, hMax]
        Cet intervalle peut changer avec le temps, en fonctions des parametres rentrés et des signaux reçus.

Quand hMax tends vers l'infini, le rendu f(hMax) est une courbe "lissée et applatie". C'est normal car notre triangle à une
base très grande (hMax) et donc la hauteur du triangle (2/hMax) tend vers 0. Si hMax est vriament très grand et englobe
l'ensemble de définition, alors on a un f(hMax) valant une constante liée à la pondération de chaque point et de la
hauteur de notre triangle.
Pour le Kernel triangulaire, pour le Kernel ellipsoidal également (voir pour les autres mais ça doit être un fait général),
à chaque fois la mesure est faite de la manière suivante : pour un point du signal entrant, on mesure en ce point la
hauteur entre ce point et la borne la plus proche (perpendiculairement donc) de figure en fonction du Kernel utilisé.
De ce fait :


    ° Pour donner une borne supérieure à notre hMax, une idée pourrait être de conserver une hauteur centrale minimale.
    Pour le Kernel triangulaire : la hauteur du triangle vaut 2/hMax, et donc en imposant une hauteur minimale (une valeur
    donnée en parametre ou une constante petite de base, si aucun parametre n'est donné), on impose une borne sup à hMax.
    Ainsi : hMax<= 2/Hauteur_minimale_souhaitée (pour le Kernel Triangulaire).

    ° Ce hMax pourrait rester trop grand. Une idée pourrait être d'étudier l'écart entre n pas (donné en parametre) du
    max(f(hMax)) et min(f(hMin)). Si sur ces n pas : |max(f(hMin)) - min(f(hMin))| < borne_donnee_en_parametre,
    alors on diminue la borne supérieure de hMax d'un epsilon (donné en parametre).


- Epsilon :

Pour les valeurs des epsilon (un pour hMin et un pour hMax), il sera proposées une petite valeur par défaut.
Cependant il faudra voir comment optimiser cette valeur afin de faire le moins de changements possibles sur hMin et hMax,
tout en garantissant le fait de conserver hOpt dans notre intervalle.
Une idée pourrait être de déployer une série de test statistiques sur des fonctions, qui ne sont pas des fonctions de bases,
et que nous pourront simuler afin de voir par la suite si il existe des epsilon qui mêlent efficacité et garantie.


- Précision sur la hauteur à conserver :

Nos tests ont été réalisés avec le Kernel triangulaire, ainsi on parle de hauteur max ou min, en parlant de la hauteur
du triangle dont la base est positionnée sur l'axe des abcisses.

Dans le cadre de l'application d'autres Kernel où la figure (triangle, ellipse,...) est symétrique, la hauteur correspond
à l'écart entre l'axe des abcisses et le point au milieu du Kernel.

Nous avons pour l'instant étudié le cas du Kernel Triangulaire qui ne fait pas intervenir d'autres parametres que notre
fenêtre détude h C [hMin,hMax].


###########################################################################################


Dans le cadre de notre étude, après implémentation des fonctions décrites ci-dessus, nous pourrons comparer nos bornes
hMin et hMax avec la valeur exacte de hOpt pour les cas connus. Il sera intéressant notamment de voir au bout de combien
de temps/d'itération, notre intervalle [hMin,hMax] encadrera de façon précise ce hOpt, en fonction du Epsilon donné.