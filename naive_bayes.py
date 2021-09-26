# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA

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

#PCA fitting and transform
pca_model = PCA(n_components = 2)
pca_model.fit(X)
print("Explained variance sums to {:.2f}".format(100*np.sum(pca_model.explained_variance_ratio_)))
X_PCA = pca_model.transform(X)


i = 0
n = 5
kfold = StratifiedKFold(n_splits = n, shuffle = True)
accuracy = np.zeros([2, n])

for train, test in kfold.split(X, Y):
    
    # RBF kernel SVM was chosen as the best. Ideally this would be picked based on
    # the validation loss, but this should work fine for comparison's sake
    rbf = svm.SVC(kernel = 'rbf', C = 1)
    rbf.fit(X[train], Y[train])
    
    # Let's see how a Naïve Bayers holds against this
    bayes = GaussianNB()
    bayes.fit(X[train], Y[train])
    
    # predictions
    y_pred_rbf = rbf.predict(X[test])
    y_pred_bayes = bayes.predict(X[test])
    
    # registering the accuracies
    accuracy[0, i] = accuracy_score(Y[test], y_pred_rbf)
    accuracy[1, i] = accuracy_score(Y[test], y_pred_bayes)
    
    # index to iterate over the accuracy matrix
    i+=1

print("Accuracy of the RBF Kernel SVM = {:.2f}".format(100*np.mean(accuracy[0,:])))
print("Accuracy of the Naïve Bayes SVM = {:.2f}".format(100*np.mean(accuracy[1,:])))

# confmat_rbf = confusion_matrix(y_test, y_pred_rbf)
# confmat_bayes = confusion_matrix(y_test, y_pred_bayes)
