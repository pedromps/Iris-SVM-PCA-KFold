# Iris Data classification with SVM and Stratified K-Fold after dimensionality reduction with PCA
This repository contains code implemented with the sklearn library for SVM classifiers with different kernels and different margins (from softer to harder margins) in the file svm.py. There is an implementation comparing an SVM classifier with a Naïve Bayes classifier in the file naive_bayes.py.


The data had its dimensionality reduced by using a Principal Component Analysis (PCA) such that the 2 features containing the most information are the ones used for classification. This was done for the purpose or later being able to plot the data in 2D (as the files in this repository show) and to take a more optimal approach than to "just pick 2 random features to train and test" the SVM classifiers. For the sake of consistency, as the dataset is small, k-fold cross-validation was used. 


Support Vector Machines (SVM) are a supervised classifier which is commonly used for the classification of data, such as the Iris dataset's. There are several kernels, which transform data and may allow for better results in the classification task. This project was made as the contact I had had with SVM previously had no included handling this famous dataset and I wanted to try it on my own.


The Naïve Bayes classifier is a probabilistic classifier which works with a maximum-likelihood estimation. It works with conditional probability (Bayesian probability) and it is called naïve as it assumes that all features are mutually independent.


The PCA is a useful tool to represent a set of many features in a smaller set, with some restrictions of course (it shouldn't be used for categorical features, for example). Using them to reduce dimensionality of datasets comes at a potential cost of losing information: in the case of this project, reducing from 4->2 dimensions loses around 5% of information. Generalising this concept for datasets much larger than othis one, it is more clear that this method, while potentially losing information, can yield significant gains in computational times.


While the k-fold cross-validation could have been used in this work with no issue, the stratified k-fold cross-validation was chosen instead, as it ensures each fold has the same representation of each class as the original dataset. This is more crucial to use in datasets with class imbalance (which is not the case here as each of the 3 classes has the same representation), however using it makes the results have less variance across the folds.


