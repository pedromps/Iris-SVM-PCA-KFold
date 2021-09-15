# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

#functions for decision boundaries, taken from https://stackoverflow.com/questions/51495819/how-to-plot-svm-decision-boundary-in-sklearn-python
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    Z[np.where(Z == "Iris-setosa")]=255
    Z[np.where(Z == "Iris-versicolor")]=0
    Z[np.where(Z == "Iris-virginica")]=120
    out = ax.contourf(xx, yy, Z, **params)
    return out


#import data
iris_data = pd.read_csv(os.path.join(os.getcwd(), "data/iris.data"))

#separate labels and data
Y = np.array(iris_data["5"])
ivs = np.where(Y=="Iris-setosa")
ive = np.where(Y=="Iris-versicolor")
ivi = np.where(Y=="Iris-virginica")


X = np.array(iris_data[["1", "2", "3", "4"]])
#normalise data between 0 and 1
X = (X-np.nanmin(X, axis=0))/(np.nanmax(X, axis=0)-np.nanmin(X, axis=0))


#if the PCA has a good explained variance ratio, we can reduce dimensions without losing much info
pca = PCA(copy=True, iterated_power='auto', n_components=2, random_state=None, svd_solver='auto', tol=0.0, whiten=False)
pca_input = X
pca = pca.fit(pca_input) 
print(pca.explained_variance_ratio_)
Z_pca = pca.transform(pca_input)
plt.figure()
plt.title("PCA Latent Space Representation and Reconstruction with two components where the % explained variance of each component is $c_1$ = {:.2f}% and $c_2$ = {:.2f}%"\
          .format(pca.explained_variance_ratio_[0]*100, pca.explained_variance_ratio_[1]*100))
plt.scatter(*Z_pca[ivs].T, c='blue')
plt.scatter(*Z_pca[ive].T, c='yellow')
plt.scatter(*Z_pca[ivi].T, c='red')
plt.legend(['Iris Setosa', 'Iris Versicolor', 'Iris Virginica'], loc='best')
plt.grid()
plt.xlabel('$c_1$')
plt.ylabel('$c_2$')
plt.show()

info = pca.explained_variance_ratio_[0]*100+pca.explained_variance_ratio_[1]*100
print("We maintain a total of {:.2f}% of the information".format(info))


#k fold
n = 5
kfold = KFold(n_splits=n)
accuracy = np.zeros(n)
i=0
for train, test in kfold.split(X, Y):
    
    #lowering the dimension of data so that it is 2d
    pca_model = PCA(info/100)
    pca_model.fit(X[train])
    x_train = pca_model.transform(X[train])
    x_test = pca_model.transform(X[test])
    
    #I wanna test more kernels later
    clf = svm.SVC()
    clf.fit(x_train, Y[train])
    
    y_pred = clf.predict(x_test)
    
    accuracy[i] = accuracy_score(y_pred, Y[test])
    i+=1
    y_train = Y[train]
    
#plotting the last split:
fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface of SVC ')
# Set-up grid for plotting.
X0, X1 = x_train[:, 0], x_train[:, 1]
xx, yy = make_meshgrid(X0, X1)

y_train[np.where(y_train == "Iris-setosa")]='r'
y_train[np.where(y_train == "Iris-versicolor")]='g'
y_train[np.where(y_train == "Iris-virginica")]='b'

plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_ylabel('feature 1')
ax.set_xlabel('feature 2')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
plt.show()