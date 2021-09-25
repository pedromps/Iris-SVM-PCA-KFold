# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


# functions for decision boundaries, taken from https://stackoverflow.com/questions/51495819/how-to-plot-svm-decision-boundary-in-sklearn-python
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


# import data
iris_data = pd.read_csv(os.path.join(os.getcwd(), "data/iris.data"))

# separate labels and data
Y = np.array(iris_data["5"])
ivs = np.where(Y == "Iris-setosa")
ive = np.where(Y == "Iris-versicolor")
ivi = np.where(Y == "Iris-virginica")


X = np.array(iris_data[["1", "2", "3", "4"]])
# normalise data between 0 and 1, it's advised
X = (X-np.nanmin(X, axis = 0))/(np.nanmax(X, axis = 0)-np.nanmin(X, axis = 0))


# if the PCA has a good explained variance ratio, we can reduce dimensions without losing much info
pca = PCA(copy = True, iterated_power = 'auto', n_components = 2, random_state = None, svd_solver = 'auto', tol = 0.0, whiten = False)
pca_input = X
pca = pca.fit(pca_input) 
print(pca.explained_variance_ratio_)
Z_pca = pca.transform(pca_input)
plt.figure()
plt.title("PCA Latent Space Representation and Reconstruction with two components where the % explained variance of each component is $c_1$ = {:.2f}% and $c_2$ = {:.2f}%"\
          .format(pca.explained_variance_ratio_[0]*100, pca.explained_variance_ratio_[1]*100))
plt.scatter(*Z_pca[ivs].T, c = 'blue')
plt.scatter(*Z_pca[ive].T, c = 'yellow')
plt.scatter(*Z_pca[ivi].T, c = 'red')
plt.legend(['Iris Setosa', 'Iris Versicolor', 'Iris Virginica'], loc = 'best')
plt.grid()
plt.xlabel('$c_1$')
plt.ylabel('$c_2$')
plt.show()

info = pca.explained_variance_ratio_[0]*100+pca.explained_variance_ratio_[1]*100
print("We maintain a total of {:.2f}% of the information".format(info))


# various margins to test out 
C = [0.01, 0.1, 1, 10, 100]

for margin in C:
    # k fold
    n = 5
    # the regular kfold can be used too, but the stratified kfold is better here
    # the advantage here is that the folds preserve the percentage of samples for each class
    # the regualr kfold might yield folds where at least one class is absent, this is
    # especially problematic in situations with class imbalance (which isnt the current case)
    kfold = StratifiedKFold(n_splits = n)
    
    #4 because I'm using 4 kernels
    accuracy = np.zeros([4, n])
    i = 0
    for train, test in kfold.split(X, Y):
        
        # lowering the dimension of data so that it is 2d
        # the PCA was calculated before and it had a explained variance, so we can try to split data
        pca_model = PCA(0.99*info/100)
        pca_model.fit(X[train])
        
        x_train = pca_model.transform(X[train])
        x_test = pca_model.transform(X[test])
         
        # testing 4 kernels
        linear = svm.SVC(kernel = 'linear', C = margin)
        linear.fit(x_train, Y[train])
        
        rbf = svm.SVC(kernel = 'rbf', C = margin)
        rbf.fit(x_train, Y[train])
        
        poly = svm.SVC(kernel = 'poly', C = margin, degree = 3)
        poly.fit(x_train, Y[train])
        
        sigm = svm.SVC(kernel = 'sigmoid', C = margin)
        sigm.fit(x_train, Y[train])
        
        for j, clf in enumerate((linear, rbf, poly, sigm)):
            y_pred = clf.predict(x_test)
            accuracy[j, i] = accuracy_score(Y[test], y_pred)
            
        i+=1
        y_test= Y[test]
        
    
    y_test[np.where(y_test == "Iris-setosa")] = 'blue'
    y_test[np.where(y_test == "Iris-versicolor")] = 'yellow'
    y_test[np.where(y_test == "Iris-virginica")] = 'red'
    
    # plotting the last split, possible as the PCA reduced dimensionality from 4 to 2 while losing just 5% of information:
    fig, ax = plt.subplots(figsize=(18.64, 9.48))
    # title for the plots
    title1 = ('Decision surface for the ')
    title2 = ['Linear Kernel','RBF Kernel','Polynomial Kernel','Sigmoid Kernel']
    title3 = (' Formulation. Accuracy = ')
    # Set-up grid for plotting.
    for i, clf in enumerate((linear, rbf, poly, sigm)):
        plt.subplot(2, 2, i + 1)
        X0, X1 = x_test[:, 0], x_test[:, 1]
        xx, yy = make_meshgrid(X0, X1)
        
        # predict and colour classes
        # code adapted from 
        # https://towardsdatascience.com/multiclass-classification-with-support-vector-machines-svm-kernel-trick-kernel-functions-f9d5377d6f02
        # https://stackoverflow.com/questions/51495819/how-to-plot-svm-decision-boundary-in-sklearn-python
        aux = xx.ravel()
        auxy = yy.ravel()
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        Z[np.where(Z == "Iris-setosa")] = 255
        Z[np.where(Z == "Iris-versicolor")] = 0
        Z[np.where(Z == "Iris-virginica")] = 120
        #plot them
        plt.contourf(xx, yy, Z, cmap = plt.cm.coolwarm, alpha = 0.8)
        
        plt.scatter(X0, X1, c = y_test, cmap=plt.cm.coolwarm, s = 20, edgecolors = 'k')
        plt.ylabel('feature 1')
        plt.xlabel('feature 2')
        plt.xticks(())
        plt.yticks(())
        #accuracy also printed here
        plt.title(title1+str(title2[i])+title3+"{:.2f}%".format(np.mean(accuracy[i,:]*100)))
        plt.show()
        plt.savefig(fname = str(margin)+".png")   
        		      
