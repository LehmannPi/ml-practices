import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as sm
#from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

# =============================================================================
# Examples of non-linear functions that can be used to match the data format
# =============================================================================

x = np.arange(-5.0, 5.0, 0.1)
y = 1*(x**3) + 1*(x**2) + 1*x + 3
y_noise = 20 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo', markersize=3)
plt.plot(x,y, 'r')
plt.title("Polynomial")
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

x = np.arange(-5.0, 5.0, 0.1)
y = np.power(x,2)
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo', markersize=3)
plt.plot(x,y, 'r') 
plt.title("Quadratic")
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

X = np.arange(-5.0, 5.0, 0.1)
Y= np.exp(X)
Y_noise = 4*np.random.normal(size=X.size)
Y_data = Y + Y_noise
plt.plot(X,Y,'r')
plt.plot(X,Y_data, 'bo', markersize=3)
plt.title("Exponential")
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

X = np.arange(-5.0, 5.0, 0.1)
Y = np.log(X)
Y_noise = 0.4*np.random.normal(size=X.size)
Y_data = Y + Y_noise
plt.plot(X,Y,'r')
plt.plot(X,Y_data,'bo',markersize=3)
plt.title("Logarithmic")
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

X = np.arange(-5.0, 8.0, 0.1)
Y = 1+4/(1+np.power(3, X-2))
Y_noise = 0.5*np.random.normal(size=X.size)
Y_data = Y + Y_noise
plt.plot(X,Y_data,'bo', markersize=3)
plt.plot(X,Y,'r')
plt.title("Logistic/Sigmoidal")
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

# =============================================================================
# Starting regression process
# =============================================================================

# The dataset
df = pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/china_gdp.csv")
df.head()

plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)
plt.plot(x_data, y_data, 'ro', markersize=4)
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

# Choosing a model - The logistic could be a good aproximation because the rise
# of the curve is not very sudden and the growth decreases at the end
X = np.arange(-5.0, 5.0, 0.1)
Y = 1.0 / (1.0 + np.exp(-X))
plt.plot(X,Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

# With the logistic function at hand, the model can be build.
# Function: y = 1 / ( 1+ exp^(Beta1*(X-Beta2)))
# Beta1: Curve's steepness
# Beta2: Slide curve on x-axis

def sigmoid(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(Beta_1*(x-Beta_2)))
     return y

# Normalizing the data, as to enable the function fitting
xdata = x_data/max(x_data)
ydata = y_data/max(y_data)

# The coefficients
popt, pcov = curve_fit(sigmoid, xdata, ydata)
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

# Calculating the accuracy of the model
msk = np.random.rand(len(df))<0.8
train_x = xdata[msk]
test_x = xdata[~msk]
train_y = ydata[msk]
test_y = ydata[~msk]

popt, pcov = curve_fit(sigmoid,train_x,train_y)
y_hat = sigmoid(test_x, *popt)
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
print("Residual sum of squares (MSE): %.4f" % np.mean((y_hat - test_y)**2))
print("R2-score: %.2f" % sm.r2_score(y_hat, test_y))

print("Coeficient values:", popt) # Predicted OPTimal values
print("Estimated covariance (array):\n", pcov) # Predicted COVariance

