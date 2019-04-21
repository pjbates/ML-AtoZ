# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('c:\src\ML-AtoZ\Part 1 - Data Preprocessing\Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#Cleaning Data - Using the new functions. 
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values = np.nan, strategy = 'mean')
X[:, 1:3] = imp_mean.fit_transform(X[:, 1:3])

#Encode Category Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_X = LabelEncoder()
X[:, 0] = le_X.fit_transform(X[:, 0])
ohe_X = OneHotEncoder(categorical_features = [0])
X = ohe_X.fit_transform(X).toarray()

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
process = make_column_transformer(
    (OneHotEncoder(), [0]),
    remainder="passthrough"
    )

A = process.fit_transform(X)





# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""