import pandas as pd

df = pd.read_csv('iris.data')

import matplotlib.pyplot as plt
import numpy as np

#selecting only sitosa and versicolor (binary classifier)
y = df.iloc[0:100, 4].values
#Giving numerical labels to categorical labels
y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[0:100, [0,2]].values

#plot data
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upperleft')
plt.show()