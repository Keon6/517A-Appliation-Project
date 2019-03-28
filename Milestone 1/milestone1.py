
##########################################
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
IDs = pd.read_csv("../input/test.csv")["ID"]
##########################################


##########################################
fire = data.copy()
fire['elu_cli'] = [int(str(a)[0]) for a in fire['Soil_Type']]
fire['elu_geo'] = [int(str(a)[1]) for a in fire['Soil_Type']]
fire['elu_3'] = [int(str(a)[2]) for a in fire['Soil_Type']]
fire['elu_4'] = [int(str(a)[3]) for a in fire['Soil_Type']]

X_test = test_data.copy()
X_test['elu_cli'] = [int(str(a)[0]) for a in X_test['Soil_Type']]
X_test['elu_geo'] = [int(str(a)[1]) for a in X_test['Soil_Type']]
X_test['elu_3'] = [int(str(a)[2]) for a in X_test['Soil_Type']]
X_test['elu_4'] = [int(str(a)[3]) for a in X_test['Soil_Type']]

fireX = fire.drop(['Horizontal_Distance_To_Fire_Points', 'Soil_Type', "ID"], axis=1)  # takes out Y values from dataset
fireY = fire['Horizontal_Distance_To_Fire_Points'].copy()  # only the Y values

X_test = X_test.drop(["ID", "Soil_Type"], axis=1)
##########################################


##########################################
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


cat_attribs = ['elu_cli', 'elu_geo', 'elu_3', 'elu_4']
fire_num = fireX.drop(cat_attribs, axis=1)
num_attribs = list(fire_num)

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('std_scaler', StandardScaler())
])
encoder = OneHotEncoder()
cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('encoder', encoder),
])
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])
fireX = full_pipeline.fit_transform(fireX).toarray()
fireX = pd.DataFrame(fireX, columns=list(fire_num.columns) + list(range(20)))  # retain names of columns
##########################################

##########################################
test_num = X_test.drop(cat_attribs, axis=1)
num_attribs = list(test_num)
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('std_scaler', StandardScaler())
])
encoder = OneHotEncoder()
cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('encoder', encoder),
])
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])
X_test = full_pipeline.fit_transform(X_test).toarray()
X_test = pd.DataFrame(X_test, columns=list(test_num.columns) + list(range(20)))  # retain names of columns
##########################################

##########################################
# IDEA:
# 1. RUN Kernel PCA (try RBF and Quadratic)
# 2. (In R) RUN Linear Regression on principal component data &evaluate model assumptions
# 3. k-fold CV with Generalized Linear Model /Lin Reg (depending on whether Guass-Markov Assumptions hold or not)
# 4. Hypothesis Test for RBF Kernal vs. Quadratic Kernel
##########################################

##########################################
print(fireX.head())
print(X_test.head())
print(fireX.shape)
print(X_test.shape)
##########################################


##########################################
# kernal PCA
from sklearn.decomposition import KernelPCA

# Quadratic Kernal
quad_pca = KernelPCA(kernel="poly", fit_inverse_transform=True, gamma=2, coef0=1)
X_quad_pca = quad_pca.fit_transform(fireX)

quad_pca = KernelPCA(kernel="poly", fit_inverse_transform=True, gamma=2, coef0=1, n_components=X_quad_pca.shape[1])
X_test_quad = quad_pca.fit_transform(X_test)

# RBF Kernel
rbf_pca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=1)
X_rbf_pca = rbf_pca.fit_transform(fireX)

rbf_pca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=1, n_components=X_rbf_pca.shape[1])
X_test_rbf = rbf_pca.fit_transform(X_test)
##########################################

##########################################
print(X_quad_pca.shape)
print(X_quad_pca[0:3, ])
print(X_rbf_pca.shape)
print(X_rbf_pca[0:3, ])

print(X_test_quad.shape)
print(X_test_quad[0:3, ])
print(X_test_rbf.shape)
print(X_test_rbf[0:3, ])
##########################################

##########################################
# IDEA:
# FROM R, it was shown that there are lots of outliers & non-normal residuals
# (Residuals were fat-tailed...)
# TRY ROBUST REGRESSIONS
from sklearn.linear_model import LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

Y = np.asarray(fireY)
IDs = np.asarray(IDs)
IDs = IDs.reshape(len(IDs), 1)
OLS_Estimator = LinearRegression()
Huber_Estimator = HuberRegressor()
Ridge_Estimator = KernelRidge(alpha=0.002, kernel="rbf", gamma=1)
# estimators = [('OLS', LinearRegression()),
#               ('Huber', HuberRegressor())]
# ('RANSAC', RANSACRegressor(random_state=42, min_samples=6000))
# ('Theil-Sen', TheilSenRegressor(random_state=42, n_jobs=10))

# CV Prep
n, k = X_quad_pca.shape[0], 10
val_size = n // k
##########################################

##########################################
# 1. RBF kPCA + OLS
model1 = OLS_Estimator.fit(X_rbf_pca, Y)
train_fit = model1.predict(X_rbf_pca)
rmse1 = np.sqrt(mean_squared_error(train_fit, Y))
print(rmse1)  # RMSE =  1.2875957631480734e-11
test_pred1 = model1.predict(X_test_rbf)
test_pred1 = test_pred1.reshape(len(test_pred1), 1)
test_pred1 = np.concatenate((IDs, test_pred1), axis=1)
np.savetxt(fname="rbf_kpca_ols.csv", X=test_pred1, delimiter=",")
##########################################

##########################################
# 2. RBF Ridge Reg alpha = 0.002 & gamma = 1
model2 = Ridge_Estimator.fit(fireX, Y)
train_fit = model2.predict(fireX)
rmse2 = np.sqrt(mean_squared_error(train_fit, Y))
print(rmse2)  # 16.64422326574506
test_pred2 = model2.predict(X_test)
test_pred2 = test_pred2.reshape(len(test_pred2), 1)
test_pred2 = np.concatenate((IDs, test_pred2), axis=1)
np.savetxt("rbf_ridge.csv", test_pred2, delimiter=",")
##########################################

##########################################
# 2. RBF Ridge Reg alpha = 0.002 & gamma = 1
model2 = Ridge_Estimator.fit(fireX, Y)
train_fit = model2.predict(fireX)
rmse2 = np.sqrt(mean_squared_error(train_fit, Y))
print(rmse2)  # 16.64422326574506
test_pred2 = model2.predict(X_test)
test_pred2 = test_pred2.reshape(len(test_pred2), 1)
test_pred2 = np.concatenate((IDs, test_pred2), axis=1)
np.savetxt("rbf_ridge.csv", test_pred2, delimiter=",")
##########################################

##########################################
# 3. RBF kPCA + Huber
model3 = Huber_Estimator.fit(X_rbf_pca, Y)
train_fit = model3.predict(X_rbf_pca)
rmse3 = np.sqrt(mean_squared_error(train_fit, Y))
print(rmse3)
test_pred3 = model3.predict(X_test_rbf)
test_pred3 = test_pred3.reshape(len(test_pred3), 1)
test_pred3 = np.concatenate((IDs, test_pred3), axis=1)
np.savetxt("rbf_kpca_huber.csv", test_pred3, delimiter=",")
##########################################

##########################################
# 4. Quad kPCA + Huber
model4 = Huber_Estimator.fit(X_quad_pca, Y)
train_fit = model4.predict(X_quad_pca)
rmse4 = np.sqrt(mean_squared_error(train_fit, Y))
print(rmse4)
test_pred4 = model4.predict(X_test_quad)
test_pred4 = test_pred4.reshape(len(test_pred4), 1)
test_pred4 = np.concatenate((IDs, test_pred4), axis=1)
np.savetxt("quad_kpca_huber.csv", test_pred4, delimiter=",")
##########################################

##########################################
# 5. Quad kPCA + OLS
model5 = OLS_Estimator.fit(X_quad_pca, Y)
train_fit = model5.predict(X_quad_pca)
rmse5 = np.sqrt(mean_squared_error(train_fit, Y))
print(rmse5)
test_pred5 = model5.predict(X_test_quad)
test_pred5 = test_pred5.reshape(len(test_pred5), 1)
test_pred5 = np.concatenate((IDs, test_pred5), axis=1)
np.savetxt("quad_kpca_ols.csv", test_pred5, delimiter=",")
##########################################

#  # for quadratic kernel
# rmses_q = dict()
# for name, estimator in estimators:
#     rmse_cv = 0
#     rmses_q[name] = []
#     for i in range(k):
#         X_cv, Y_cv = X_quad_pca[i * val_size:(i + 1) * val_size,:], Y[i * val_size:(i + 1) * val_size]
#         train_indices = list(range(0, i * val_size)) + list(range((i + 1) * val_size, n))
#         X_tr, Y_tr = X_quad_pca[train_indices,:], Y[train_indices]

#         model = estimator.fit(X_tr, Y_tr)
#         fits = model.predict(X_tr)
#         mn, mx = min(fits), max(fits)
#         if mn<0: mn=0
#         pred = model.predict(X_cv)
#         pred[np.where(pred > mx)] = mx
#         pred[np.where(pred < mn)] = mn
#         rmse_v = np.sqrt(mean_squared_error(pred, Y_cv))
#         rmses_q[name].append(rmse_v)
#         rmse_cv += rmse_v
#     rmse_cv = rmse_cv/k
#     print(name, "CV RMSE: ", round(rmse_cv,5))
# #         weights.append(model.coef_ )

# # weights = np.asarray(weights)
# # np.savetxt("weights_quad_kernel.csv", weights, delimiter=",")
​
# # QUADRATIC KERNEL TRAINING RMSE
# # OLS 322.86766
# # Huber 526.68547
​
# # OLS CV RMSE:  843.47723
# # Huber CV RMSE:  564.21689
# for rbf kernel
# rmses = dict()
# for name, estimator in estimators:
#     rmse_cv = 0
#     rmses[name] = []
#     for i in range(k):
#         X_cv, Y_cv = X_rbf_pca[i * val_size:(i + 1) * val_size,:], Y[i * val_size:(i + 1) * val_size]
#         train_indices = list(range(0, i * val_size)) + list(range((i + 1) * val_size, n))
#         X_tr, Y_tr = X_rbf_pca[train_indices,:], Y[train_indices]


#         model = estimator.fit(X_tr, Y_tr)
#         fits = model.predict(X_tr)
#         mn, mx = min(fits), max(fits)
#         if mn<0: mn=0
#         pred = model.predict(X_cv)
#         pred[np.where(pred > mx)] = mx
#         pred[np.where(pred < mn)] = mn

#         rmse_v = np.sqrt(mean_squared_error(pred, Y_cv))
#         rmses[name].append(rmse_v)
#         rmse_cv += rmse_v
#     rmse_cv = rmse_cv/k
#     print(name, "CV RMSE: ", round(rmse_cv,5))

# # RBF TRAINING RMSE
# # OLS 0.0
# # Huber 233.44207

# # CV RMSE
# # OLS CV RMSE:  370.74515
# # Huber CV RMSE:  446.9084
# # for rbf kernel
# rmses = dict()
# for name, estimator in estimators:
#     rmse_cv = 0
#     rmses[name] = []
#     for i in range(k):
#         X_cv, Y_cv = X_rbf_pca[i * val_size:(i + 1) * val_size,:], Y[i * val_size:(i + 1) * val_size]
#         train_indices = list(range(0, i * val_size)) + list(range((i + 1) * val_size, n))
#         X_tr, Y_tr = X_rbf_pca[train_indices,:], Y[train_indices]


#         model = estimator.fit(X_tr, Y_tr)
#         fits = model.predict(X_tr)
#         mn, mx = min(fits), max(fits)
#         if mn<0: mn=0
#         pred = model.predict(X_cv)
#         pred[np.where(pred > mx)] = mx
#         pred[np.where(pred < mn)] = mn

#         rmse_v = np.sqrt(mean_squared_error(pred, Y_cv))
#         rmses[name].append(rmse_v)
#         rmse_cv += rmse_v
#     rmse_cv = rmse_cv/k
#     print(name, "CV RMSE: ", round(rmse_cv,5))
​
# # RBF TRAINING RMSE
# # OLS 0.0
# # Huber 233.44207
​
# # CV RMSE
# # OLS CV RMSE:  370.74515
# # Huber CV RMSE:  446.9084
# RANSAC
# estimator = RANSACRegressor(random_state=42, min_samples=6000)
# rmse_cv_rbf = [0, 0]
# rmses_ransac = dict()
# rmses_ransac["RBF"] = []
# rmses_ransac["QUAD"] = []

# for i in range(k):
#         X_cv, Y_cv = X_rbf_pca[i * val_size:(i + 1) * val_size,:], Y[i * val_size:(i + 1) * val_size]
#         train_indices = list(range(0, i * val_size)) + list(range((i + 1) * val_size, n))
#         X_tr, Y_tr = X_rbf_pca[train_indices,:], Y[train_indices]
#         model = estimator.fit(X_tr, Y_tr)
#         fits = model.predict(X_tr)
#         mn, mx = min(fits), max(fits)
#         if mn<0: mn=0
#         pred = model.predict(X_cv)
#         pred[np.where(pred > mx)] = mx
#         pred[np.where(pred < mn)] = mn
#         rmse_v = np.sqrt(mean_squared_error(pred, Y_cv))
#         rmses_ransac["RBF"].append(rmse_v)
#         rmse_cv_rbf[0] += rmse_v

#         X_cv = X_quad_pca[i * val_size:(i + 1) * val_size,:]
#         X_tr = X_quad_pca[train_indices,:]
#         model = estimator.fit(X_tr, Y_tr)
#         fits = model.predict(X_tr)
#         mn, mx = min(fits), max(fits)
#         if mn<0: mn=0
#         pred = model.predict(X_cv)
#         pred[np.where(pred > mx)] = mx
#         pred[np.where(pred < mn)] = mn
#         rmse_v = np.sqrt(mean_squared_error(pred, Y_cv))
#         rmses_ransac["QUAD"].append(rmse_v)
#         rmse_cv_rbf[1] += rmse_v
# rmse_cv = rmse_cv/k
# print("CV RMSE: ", round(rmse_cv[0],5), round(rmse_cv[1],5))
# # RANSAC
# estimator = RANSACRegressor(random_state=42, min_samples=6000)
# rmse_cv_rbf = [0, 0]
# rmses_ransac = dict()
# rmses_ransac["RBF"] = []
# rmses_ransac["QUAD"] = []
​
# for i in range(k):
#         X_cv, Y_cv = X_rbf_pca[i * val_size:(i + 1) * val_size,:], Y[i * val_size:(i + 1) * val_size]
#         train_indices = list(range(0, i * val_size)) + list(range((i + 1) * val_size, n))
#         X_tr, Y_tr = X_rbf_pca[train_indices,:], Y[train_indices]
#         model = estimator.fit(X_tr, Y_tr)
#         fits = model.predict(X_tr)
#         mn, mx = min(fits), max(fits)
#         if mn<0: mn=0
#         pred = model.predict(X_cv)
#         pred[np.where(pred > mx)] = mx
#         pred[np.where(pred < mn)] = mn
#         rmse_v = np.sqrt(mean_squared_error(pred, Y_cv))
#         rmses_ransac["RBF"].append(rmse_v)
#         rmse_cv_rbf[0] += rmse_v

#         X_cv = X_quad_pca[i * val_size:(i + 1) * val_size,:]
#         X_tr = X_quad_pca[train_indices,:]
#         model = estimator.fit(X_tr, Y_tr)
#         fits = model.predict(X_tr)
#         mn, mx = min(fits), max(fits)
#         if mn<0: mn=0
#         pred = model.predict(X_cv)
#         pred[np.where(pred > mx)] = mx
#         pred[np.where(pred < mn)] = mn
#         rmse_v = np.sqrt(mean_squared_error(pred, Y_cv))
#         rmses_ransac["QUAD"].append(rmse_v)
#         rmse_cv_rbf[1] += rmse_v
# rmse_cv = rmse_cv/k
# print("CV RMSE: ", round(rmse_cv[0],5), round(rmse_cv[1],5))

rmse = {0.125: [285.03372125675617,
  275.76407597620636,
  250.33125257650119,
  244.9122455043331,
  272.922473457007,
  357.99559929410685,
  258.9556644333368,
  280.9406807509435,
  253.38545934492296,
  253.74772445252546],
 0.15: [276.1574866855687,
  271.31555911255356,
  247.6675998493432,
  239.37370166816126,
  267.7406182875143,
  351.6926877928224,
  253.73645146487814,
  277.43044055551724,
  248.53773490699712,
  252.5152755777711],
 0.175: [270.39755821912127,
  269.97391786586746,
  248.67015110783865,
  235.11063515697134,
  265.28174744505407,
  347.0715022653141,
  249.6942692035739,
  276.7614037268522,
  248.68996332713672,
  255.05611358306595],
 0.2: [266.79052751881085,
  269.8974335151598,
  251.03518146264437,
  231.74590644262057,
  263.3906563835122,
  343.3215573535837,
  245.57102928361115,
  277.2209191260148,
  252.61232999557674,
  257.81573764452503],
 0.225: [265.0458380019343,
  270.2040206228356,
  253.98338997870093,
  229.10808299767763,
  261.34265364421765,
  339.9315216137944,
  241.39922542281238,
  277.93703682669724,
  258.68111234888306,
  259.6829853252715],
 0.25: [264.8876998213956,
  270.6348664990137,
  256.5207402085622,
  227.03056207910902,
  259.1232794919388,
  336.85702106495813,
  237.677554510136,
  279.0333228528277,
  266.1739651372141,
  260.8307743369048],
 0.35: [277.08927584474736,
  276.8060343287451,
  276.8952069024762,
  223.3438395466586,
  253.38523378012175,
  322.38669642349174,
  231.7330059680999,
  287.3632123212113,
  300.8268190217196,
  266.40581451564753],
 0.375: [282.8519186888747,
  280.1109245818951,
  283.45684477344645,
  223.91592924668578,
  253.75735976481454,
  321.78355682707996,
  232.12237450765056,
  290.1458135355831,
  308.14715765731546,
  269.14176069738465],
 0.4: [289.3707681110234,
  284.24140345560835,
  290.2647877163531,
  225.20769470475452,
  255.05977453755332,
  322.45446831834204,
  233.0559177179043,
  293.19802023263964,
  314.486848067227,
  272.55079376855673]}