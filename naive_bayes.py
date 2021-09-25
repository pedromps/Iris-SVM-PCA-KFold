# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

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

x_train, x_test, y_train, y_test = train_test_split(X, Y)


# RBF kernel SVM was chosen as the best. Ideally this would be picked based on
# the validation loss, but this should work fine for comparison's sake
rbf = svm.SVC(kernel = 'rbf', C = 1)
rbf.fit(x_train, y_train)

# Let's see how a Naïve Bayers holds against this
bayes = GaussianNB()
bayes.fit(x_train, y_train)

y_pred_rbf = rbf.predict(x_test)
y_pred_bayes = bayes.predict(x_test)
print("Accuracy of the RBF Kernel SVM = {:.2f}".format(100*accuracy_score(y_test, y_pred_rbf)))
print("Accuracy of the Naïve Bayes SVM = {:.2f}".format(100*accuracy_score(y_test, y_pred_bayes)))
