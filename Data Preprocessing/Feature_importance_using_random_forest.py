'''
Another useful approach to select relevant features from a dataset is to use random forest, an ensemble technique.
'''
#%%
import pandas as pd 
df = pd.read_csv('/home/tenacious/Documents/ML and AI/Pyhton Machine Learning/Data Preprocessing/wine.data', header=None)
df.columns = ['Class label', 'Alcohol', 'Malic Acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total Phenols',
'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
'Proline']

from sklearn.model_selection import train_test_split
X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

#%%
from sklearn.ensemble import RandomForestClassifier
import numpy as np
feat_labels = df.columns[1:]

forest = RandomForestClassifier(n_estimators=500, random_state=1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) % -*s %f" % (f+1, 30, feat_labels[indices[f]], importances[indices[f]]) )

import matplotlib.pyplot as plt

plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), feat_labels, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()
