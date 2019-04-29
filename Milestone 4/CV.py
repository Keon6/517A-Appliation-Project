
from rbf_tester import *
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("input"))

# Any results you write to the current directory are saved as output.

data = pd.read_csv("input/train.csv")
test_data = pd.read_csv("input/test.csv")
IDs = pd.read_csv("input/test.csv", dtype=str)["ID"]
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

from pandas.plotting import scatter_matrix # correlation plots

# attributes = ["Horizontal_Distance_To_Fire_Points", "Horizontal_Distance_To_Hydrology",
#               "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways"]
# scatter_matrix(fire[attributes], figsize=(20,12))
fireX = fire.drop(['Horizontal_Distance_To_Fire_Points', 'Soil_Type', "ID"], axis=1) # takes out Y values from dataset
fireY = fire['Horizontal_Distance_To_Fire_Points'].copy()       # only the Y values

X_test = X_test.drop(["ID", "Soil_Type"], axis=1)
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
fireX = pd.DataFrame(fireX, columns = list(fire_num.columns) + list(range(20))) # retain names of columns

y = np.asarray(fireY)
x = np.asarray(fireX)

from sklearn.linear_model import LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA, KernelPCA

from pca_tester import *
pca = PCA(11, whiten=True)
pca.fit(fireX)
print(fireX.shape)
# plt.matshow(pca.components_, cmap='viridis')
# plt.yticks([0,1,2,3,4,5,6,7,8,9,10],['0','1','2','3','4','5','6','7','8','9','10'],fontsize=10)
# plt.colorbar()
# plt.xticks(range(29),['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology',
#                       'Vertical_Distance_To_Hydrology',	'Horizontal_Distance_To_Roadways',
#                       'Hillshade_9am',	'Hillshade_Noon',	'Hillshade_3pm',
#                       '0',	'1',	'2',	'3',	'4',	'5',	'6',	'7',	'8',	'9',	'10',	'11',	'12',	'13',	'14',	'15',	'16',	'17',	'18',	'19']
# ,rotation=65,ha='left')
# plt.tight_layout()
# plt.show()
x_pca = pca.transform(fireX)
# print(np.sum(pca.explained_variance_ratio_))
# print(x_pca.shape)
# print(y)
tot = pd.concat([pd.DataFrame(data=x_pca), pd.DataFrame(data=fireY)], axis=1)
# #tot.hist(bins=15, color='steelblue', edgecolor='black', linewidth=1.0,
# #            xlabelsize=8, ylabelsize=8, grid=False)
# tot.pairplot()

attributes = ['0','1','2','3','4','5','6','7','8','9','10']
scatter_matrix(tot, figsize=(20,12))
# plt.tight_layout()
# plt.show()
# tot = np.zeros((7438, 12))
# tot[:,:-1] = x_pca
# tot[:,11] = y
# pd.plotting.andrews_curves(tot, 'Horizontal_Distance_To_Fire_Points')
# plt.figure()
plt.scatter(x_pca[:, 0],x_pca[:, 1], c =y)
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()
plt.show()

# kpca = KernelPCA(kernel="rbf", gamma=.225, n_components=11, alpha=.002)
# kpca.fit(fireX)
# x_kpca = kpca.transform(fireX)
# plt.figure()
# plt.scatter(x_kpca[:, 0],x_kpca[:, 1], c =y)
# plt.xlabel('component 1')
# plt.ylabel('component 2')
# plt.show()

Ridge_Estimator = KernelRidge(alpha=0.002, kernel="rbf", gamma=0.225)
#m = cross_val_score(Ridge_Estimator, x_kpca, y, scoring="neg_mean_squared_error", cv=10)
m = cross_val_score(Ridge_Estimator, x_pca, y, scoring="neg_mean_squared_error", cv=10)

#pca.fit_transform(fireX)
#Ridge_Estimator = KernelRidge(alpha=0.002, kernel="rbf", gamma=0.225)
#m = cross_val_score(Ridge_Estimator, pca, y, scoring="neg_mean_squared_error", cv=10)
print(np.mean(np.sqrt(-m)))



# REAL STUFF BELOW

# from sklearn.model_selection import KFold
#
# kf = KFold(n_splits=10, shuffle=True)
# mse = np.zeros((10, 2))
# r2 = np.zeros((10, 2))
# onePCA = np.zeros((10, 2))
# oneRF = np.zeros((10,2))
# # CV 10 times
# for i in range(10):
#     fold = 0
#     print(i)
#     # CV
#     for train_index, test_index in kf.split(x):
#         # Training and Test Splits
#         x_train, y_train = x[train_index], y[train_index]
#         x_test, y_test = x[test_index], y[test_index]
#         # Models, returns mean squared error
#         onePCA[fold,:] = rbf(x_train, y_train, x_test, y_test) # Get results for RBF
#         oneRF[fold,:] = pca_method(x_train, y_train, x_test, y_test)  # Get results for Random Forrest
#         print(onePCA[fold,:], " : ", oneRF[fold,:])
#         fold+=1
#     mse[i, 0] = np.mean(onePCA[:,0]) # Average CV
#     mse[i, 1] = np.mean(oneRF[:,0])  # Average CV
#     r2[i, 0] = np.mean(onePCA[:,1])  # Average CV
#     r2[i, 1] = np.mean(oneRF[:,1])  # Average CV
#
# print(mse)
# print(r2)
