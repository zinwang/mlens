import numpy as np
from pandas import DataFrame
from sklearn.metrics import f1_score
from sklearn.datasets import load_iris


def f1(y, p):
    return f1_score(y, p, average="micro")


from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data
y = data.target


from mlens.ensemble import SuperLearner
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

ensemble = SuperLearner(scorer=f1, verbose=True)
ensemble.add([RandomForestClassifier(), LogisticRegression(max_iter=3000)])
ensemble.add([LogisticRegression(max_iter=3000), SVC()])
ensemble.add_meta(SVC())
ensemble.fit(X[:75], y[:75])
preds = ensemble.predict(X[75:])

print(DataFrame(ensemble.data))
