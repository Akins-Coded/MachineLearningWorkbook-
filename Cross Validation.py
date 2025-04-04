from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X, y = iris.data, iris.target

kmn = KNeighborsClassifier(n_neighbors=3)

cv_scores = cross_val_score(kmn, X, y, cv=5)

print(f"cross validation scores: {cv_scores}")
print(f"mean CV Score: {cv_scores.mean()}")
