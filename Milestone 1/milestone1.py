# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

data = pd.read_csv("../input/train.csv")


###### EXPLORING DATA & PREPROCESSING
data.info() # all non-null integers, n = 7438, d = 12

data.head()

data.hist(bins = 10, figsize = (20,15))

data.describe()

data['Soil_Type'] # categorical?
# based on the USFS Ecological Landtype Units (ELUs) for this study area. 
# The first digit refers to the climatic zone, the second refers to the geologic. 
# The third and fourth ELU digits are unique to the mapping unit and have no special meaning to the climatic or geologic zones.

# # From ML A-Z
# X = data.iloc[:, 1:-1].values
# y = data.iloc[:, -1].values

# print(X)
# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
# scX = StandardScaler()
# X = scX.fit_transform(X)

# # fitting logistic regression to training set
# from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression(random_state = 0)
# classifier.fit(X_train, y_train)

# # predicting the test set results
# y_pred = classifier.predict(X_test)

fire = data.copy()
fire.plot(kind='scatter', x='Hillshade_9am', y='Hillshade_Noon', alpha=0.1) # alpha lets us visualize high-density areas

fire['Soil_Type']
fire['elu_cli'] = [int(str(a)[0]) for a in fire['Soil_Type']]
fire['elu_geo'] = [int(str(a)[1]) for a in fire['Soil_Type']]
fire['elu_3'] = [int(str(a)[2]) for a in fire['Soil_Type']]
fire['elu_4'] = [int(str(a)[3]) for a in fire['Soil_Type']]

# explore.plot(kind="scatter", x="Horizontal_Distance_To_Hydrology", y="Vertical_Distance_To_Hydrology",alpha=0.5,
#               c="Horizontal_Distance_To_Fire_Points", cmap=plt.get_cmap("jet"), colorbar=True
#             )

from pandas.plotting import scatter_matrix

attributes = ["Horizontal_Distance_To_Fire_Points", "Horizontal_Distance_To_Hydrology", 
              "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways"]
scatter_matrix(fire[attributes], figsize=(20,12))

# divided around Horizontal_Distance_To_Fire_Points = 3000

fire.info()

fireX = fire.drop(['Horizontal_Distance_To_Fire_Points', 'Soil_Type'], axis=1) # takes out Y values from dataset
fireY = fire['Horizontal_Distance_To_Fire_Points'].copy()       # only the Y values

type(fireX)

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
fireX.head()


###### MODEL SELECTION & TRAINING DATA
