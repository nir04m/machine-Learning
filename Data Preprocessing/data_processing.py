# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 13:08:44 2020

@author: Oghale Enwa
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

#importing dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories = 'auto'), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype = np.float)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# taking care of missing data
from sklearn.impute import SimpleImputer
dataset = dataset.reset_index()
imputer = SimpleImputer(missing_values = np.nan , strategy = 'mean', fill_value = None)
imputer = imputer.fit(X[:,1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

#not in template

# Encoding the Dependent variable
"""from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)"""

# Encoding the Independent variable
"""from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))"""

# Feature Scaling2from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
""""""