
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

attributes = ["Horizontal_Distance_To_Fire_Points", "Horizontal_Distance_To_Hydrology",
              "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways"]
scatter_matrix(fire[attributes], figsize=(20,12))
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

from rf_tester import *
# Ridge_Estimator = KernelRidge(alpha=0.002, kernel="rbf", gamma=0.225)
# m = cross_val_score(Ridge_Estimator, fireX, y, scoring="neg_mean_squared_error", cv=10)
# print(np.mean(np.sqrt(-m)))
# print("next")

from sklearn.model_selection import KFold

kf = KFold(n_splits=10, shuffle=True)
res = np.zeros((10, 2))
oneRBF = np.zeros(10)
oneRF = np.zeros(10)
# CV 10 times
for i in range(10):
    fold = 0
    
    # CV
    for train_index, test_index in kf.split(x):
        # Training and Test Splits
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]
        # Models, returns mean squared error
        oneRBF[fold] = rbf(x_train, y_train, x_test, y_test) # Get results for RBF
        oneRF[fold] = rf(x_train, y_train, x_test, y_test)  # Get results for Random Forrest
        fold+=1
    res[i, 0] = np.mean(oneRBF) # Average CV
    res[i, 1] = np.mean(oneRF)  # Average CV
print(res)
