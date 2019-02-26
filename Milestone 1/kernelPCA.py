import numpy as np
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA
from numpy import genfromtxt

X_train = genfromtxt("transformed.csv", delimiter=',', dtype=np.float64)[1:, 1:]
Y_train = genfromtxt("train.csv", delimiter=',', dtype=np.float64)[1:, -1]


# Kernels:
# 1. Quadratic (x+1)^2
quad_pca = KernelPCA(kernel="poly", fit_inverse_transform=True, gamma=2, coef0=1)
X_quad_pca = quad_pca.fit_transform(X_train)

# 2. RBF
rbf_pca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=2)
X_rbf_pca = rbf_pca.fit_transform(X_train)
# 3.