from sklearn.feature_selection import RFECV
import numpy as np
from numpy import genfromtxt
from sklearn.svm import SVC


dataset = genfromtxt("../dataset/full_dataset_all_stats_features.csv", delimiter=',')
X = dataset[:, 0:80]
Y = dataset[:, 80:81]

svc = SVC(kernel="linear", C=1)
rfecv = RFECV(estimator=svc, min_features_to_select=30, n_jobs=16)
rfecv.fit(X, Y)
X_rfe = rfecv.transform(X)

print(X_rfe)

dataset = np.append(X_rfe, Y, axis=1)

np.savetxt("rfe_dataset.csv", dataset, delimiter=',')
