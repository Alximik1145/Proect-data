from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train , X_test, y_train, y_test = train_test_split(iris.data, iris.target, train_size=0.25, random_state=42)

tpot = TPOTClassifier(generations=5,pupulation_size=50, verbosity=2,random_state=42)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_iris_pipeline.py')
