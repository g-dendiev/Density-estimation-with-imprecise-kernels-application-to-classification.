from sklearn import datasets
iris = datasets.load_iris()
data = iris.data # Pour un accès plus rapide
target = iris.target # Les labels associés à chaque enregistrement
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(data, target) # On aurait aussi pu utiliser le dataframe df
result = clf.predict(data)
errors = sum(result != target) # 6 erreurs sur 150 mesures
print( "Pourcentage de prédiction juste:", (150-errors)*100/150)   # 96 % de réussite