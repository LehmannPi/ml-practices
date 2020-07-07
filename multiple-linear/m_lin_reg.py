# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 19:04:35 2020

@author: filip
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model


df = pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv")

# looking at the dataset
print(df.head())
print(df.columns)

# selecting features
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(cdf.head(9))

# plotting emission values with respect to engine size
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# splitting dataset randomly into train/test
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

print('Train data.')
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue', s=15)
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
print('Test data.')
plt.scatter(test.ENGINESIZE, test.CO2EMISSIONS,  color='red', s=15)
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# =============================================================================
# Training and comparing models with different selected covariates
# =============================================================================

regr = linear_model.LinearRegression()
regr2 = linear_model.LinearRegression()

# =============================================================================
# Model using FUELCONSUMPTION_COMB
# =============================================================================
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)
# sample of the coefficients for the FUELCONSUMPTION_COMB case
print('Coefficients: ', regr.coef_)

y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print('Model w/ FUELCONSUMPTION_COMB')
print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y) ** 2))
# Note on variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))

# =============================================================================
# Model using FUELCONSUMPTION_HWY and FUELCONSUMPTION_CITY
# =============================================================================
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_HWY','FUELCONSUMPTION_CITY']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr2.fit(x,y)
y_hat2 = regr2.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_HWY','FUELCONSUMPTION_CITY']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_HWY','FUELCONSUMPTION_CITY']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print('Model w/ FUELCONSUMPTION_HWY and FUELCONSUMPTION_CITY')
print('Residual sum of squares: %.2f' % np.mean((y_hat2 - y)**2))
print('Variance score: %.2f' % regr2.score(x,y))