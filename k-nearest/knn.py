import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

df = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/teleCust1000t.csv')
print('Number of target elements in dataset:')
# Data visualization and analysis
values = df['custcat'].value_counts()
cases = { 1:'Basic Service', 2:'E-Service', 3:'Plus Service', 4:'Total Service' }
for (k,v) in list(zip(values.index,values)):
    print(cases[k], ':', v, end='\t')

print('')
print(df.hist(column='income', bins=50))
print('Dataset features (columns):')
print(df.columns, '\n')
#plt.suptitle('Income')
plt.show()

y = df['custcat'].values
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)

# Normalizing data
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
# Test/train split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape, '\n')

# Training and prediction for k=4
k = 4
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)

yhat = neigh.predict(X_test)
print("5 first predicted elements: ", yhat[0:5], '\n')
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

# =============================================================================
# Determining best k value in a range of 10
# =============================================================================

Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):

    # Training and predicting model. The acc is normalized and used as error margin in the plot
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.1)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was of", mean_acc.max(), "having k =", mean_acc.argmax()+1)
