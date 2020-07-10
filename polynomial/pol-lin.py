# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 10:24:22 2020

@author: filip
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import r2_score


df = pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv")

# basic procedures for initial data selection and visualization
print(df.head())
print(df.columns)
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue', s=15)
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# creating train/test dataset
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

# =============================================================================
# In this step, the desired degree of the equation is provided and the fit_transform 
# takes the x values (data) and output it from power 0 to power of the degree selected.
# =============================================================================

poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)
print('Sample of transformed array:\n', train_x_poly[:3])
# This way it's possible to use LinearRegression() function on the data
clf = linear_model.LinearRegression()
train_y_ = clf.fit(train_x_poly, train_y)
# The coefficients
print ('Coefficients: ', clf.coef_)
print ('Intercept: ',clf.intercept_)

# Plotting
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue', s=15)
XX = np.arange(0.0, 10.0, 0.1)
yy = clf.intercept_[0]+ clf.coef_[0][1]*XX+ clf.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.title("Polinomial Regression (degree = 2)")
plt.show()

# Evaluation
test_x_poly = poly.fit_transform(test_x)
test_y_ = clf.predict(test_x_poly)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )

# =============================================================================
# For comparison, let's use the method described with the degree of 3
# An adaptation on the ploting variable (yy) is also necessary
# =============================================================================

poly3 = PolynomialFeatures(degree=3)
train_x_poly3 = poly3.fit_transform(train_x)
clf3 = linear_model.LinearRegression()
train_y3_ = clf3.fit(train_x_poly3, train_y)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue', s=15)
XX = np.arange(0.0, 10.0, 0.1)
yy = clf3.intercept_[0]+ clf3.coef_[0][1]*XX + clf3.coef_[0][2]*np.power(XX, 2) + clf3.coef_[0][3]*np.power(XX, 3)
plt.plot(XX, yy, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.title("Polinomial Regression (degree = 3)")
plt.show()
test_x_poly3 = poly3.fit_transform(test_x)
test_y3_ = clf3.predict(test_x_poly3)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y3_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y3_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y3_ , test_y) )
# The coefficients
print ('Coefficients: ', clf3.coef_)
print ('Intercept: ',clf3.intercept_)