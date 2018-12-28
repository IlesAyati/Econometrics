# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import statsmodels.api as sm 
from sklearn import datasets

data = datasets.load_boston()

import numpy as np
import pandas as pd

# define the data predictors as the preset feature names
df = pd.DataFrame(data.data, columns=data.feature_names)

# put the target (housing value in another dataframe) 
target = pd.DataFrame(data.target, columns=["MEDV"])

## Without a constant
X = df["RM"]
Y = target["MEDV"]
X = sm.add_constant(X)
#Note the difference in argument order

model = sm.OLS(Y,X).fit()
predictions = model.predict(X) 


model.summary()

## Two explanatory variables: Number of rooms and % lower status of population
x = df[["RM","LSTAT"]]
x = sm.add_constant(x)

model2 = sm.OLS(Y,x).fit()
predictions2 = model2.predict(x) 


model2.summary()
plt.plot(predictions2)
plt.plot(Y)
plt.show()

## SKLearn

from sklearn import linear_model
from sklearn import datasets
data = datasets.load_boston() ## loads Boston dataset from datasets library

# define the predictors 
df = pd.DataFrame(data.data, columns=data.feature_names)

# Define indep (housing value -- MEDV) in another DataFrame
target = pd.DataFrame(data.target, columns=["MEDV"])

x1 = df
y1 = target[“MEDV”]
lm = linear_model.LinearRegression()
modell = lm.fit(x1,y1) ## Note order!
predictionss = lm.predict(x1)

# Whatyp
# Whatyp2