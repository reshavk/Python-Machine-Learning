'''
Sequential Backward Selection Algorithm is a sequential feature selection algorithm used for avoiding overfitting
if the algorithm is incapable of avoiding overfitting using L1 and L2 regualrization.
SBS follows greedy algorithmic paradigm

ALGORITHM : 
SBS sequentially removes features from the full feature subset until the new feature subspace contains the 
desired number of features. In order to determine which feature to be removed criterion function J is defined
which is basically the measure of difference in performance before and after the removal of a particular feature.
In more intutive terms, at each stage we eliminate the feature that causes the least performance loss after removal.
'''
#%%
from sklearn.base import clone
from itertools import combinations
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class SBS() :
    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
    
    def fit(self, X, y) :
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim-1):
                score=self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
            
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        
        self.k_score_ = self.scores_[-1]
        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score

#%%
import pandas as pd 
df = pd.read_csv('/home/tenacious/Documents/ML and AI/Pyhton Machine Learning/Data Preprocessing/wine.data', header=None)

from sklearn.model_selection import train_test_split
X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

#%%
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of Features')
plt.grid()
plt.show()

'''
The plot produced shows the classfication accuracy produced for various no. of features considered
'''