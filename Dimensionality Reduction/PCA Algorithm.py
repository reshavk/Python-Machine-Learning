#%%
import pandas as pd
df_wine = pd.read_csv('/home/tenacious/Documents/ML and AI/Python-Machine-Learning/Dimensionality Reduction/wine.data')

from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

#%%
import numpy as np 
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues\n%s' % eigen_vals)
print('\nEigneVectors\n%s' % eigen_vecs)


'''
 Now for deciding the principal components to be used after dimensionality reduction,
 we can plot the total and explained variance for each feature and select the features
 that account for the maximum variance. 
'''
#%%
tot = sum(eigen_vals)
var_exp = [(i/tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
import matplotlib.pyplot as plt
plt.bar(range(1, 14), var_exp, alpha=0.5, align='center', label='individual explained variance')
plt.step(range(1, 14), cum_var_exp, where='mid', label='cummulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.show()

'''
Feture transformation is performed after selecting the top k eigen-vectors.
Selection of no. of eigen-vectors is done on the basis of computational efficiency and
performance tradeoff.
'''

#%%
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(key = lambda k: k[0], reverse=True)

w = np.hstack((eigen_pairs[0][1][: ,np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W : \n', w)

X_train_pca = X_train_std.dot(w)

'''
Visualizing the 124*2 dimensional matrix X_train_pca formed from 124*13 dimensional matrix
X_train_std
'''

colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train==1, 0], X_train_pca[y_train==1, 1], c=c, label=l, marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()