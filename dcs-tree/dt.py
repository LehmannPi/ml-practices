import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

df = pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/drug200.csv", delimiter=",")

print('\nInfo on the database\n', df[0:5])
print(df['Drug'].value_counts())
print(df.columns)
print('Size of data', len(df), '\n')

# =============================================================================
# Preparing the data set / encoding
# =============================================================================

X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
y = df["Drug"]

print('Before preprocessing:\n', X[0:5])

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

print('After preprocessing:\n', X[0:5])

# =============================================================================
# Example of dictionary creation for encoded values
# =============================================================================

kys = le_BP.classes_
vals = le_BP.transform(kys)
kvdict = dict(zip(kys,vals))

print(kvdict)

# =============================================================================
# Setting up the decision tree
# =============================================================================

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

print('\nShape of the sets\nTrain:\n', X_trainset.shape)
print(y_trainset.shape)
print('Test:\n', X_testset.shape)
print(y_testset.shape)

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
print('\nTree default parameters\n',drugTree)
drugTree.fit(X_trainset,y_trainset)

# =============================================================================
# Predtiction. Tree graph is better visualized at folder
# =============================================================================

predTree = drugTree.predict(X_testset)
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

dot_data = StringIO()
filename = "drugtree.png"
featureNames = df.columns[0:5]
targetNames = df["Drug"].unique().tolist()
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False,rounded=True)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(50, 100))
plt.imshow(img,interpolation='nearest')