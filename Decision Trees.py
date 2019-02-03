'''
Decision Trees
In this lab exercise, you will learn a popular machine learning algorithm, Decision Tree.
You will use this classification algorithm to build a model from historical data of patients, and their response to different medications.
Then you use the trained decision tree to predict the class of a unknown patient, or to find a proper drug for a new patient.
'''
import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
'''
About the dataset
Imagine that you are a medical researcher compiling data for a study. You have collected data about a set of patients, all of whom suffered
from the same illness. During their course of treatment, each patient responded to one of 5 medications, Drug A, Drug B, Drug c, Drug x 
and y.Part of your job is to build a model to find out which drug might be appropriate for a future patient with the same illness. 
The feature sets of this dataset are Age, Sex, Blood Pressure, and Cholesterol of patients, and the target is the drug that each 
patient responded to.It is a sample of binary classifier, and you can use the training part of the dataset to build a decision tree, 
and then use it to predict the class of a unknown patient, or to prescribe it to a new patient.
'''
!wget -O drug200.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/drug200.csv
my_data = pd.read_csv("drug200.csv", delimiter=",")
my_data[0:5]
'''
Pre-processing
Using my_data as the Drug.csv data read by pandas, declare the following variables: 

X as the Feature Matrix (data of my_data)
y as the response vector (target)
Remove the column containing the target name since it doesn't contain numeric values.
'''

X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]

'''
As you may figure out, some features in this dataset are categorical such as Sex or BP. Unfortunately, Sklearn Decision Trees do not 
handle categorical variables. But still we can convert these features to numerical values.
pandas.get_dummies() Convert categorical variable into dummy/indicator variables.
'''
from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

X[0:5]

#Now we can fill the target variable.
y = my_data["Drug"]
y[0:5]

'''
Setting up the Decision Tree
We will be using train/test split on our decision tree. Let's import train_test_split from sklearn.cross_validation.
'''
from sklearn.model_selection import train_test_split
'''
Now train_test_split will return 4 different parameters. We will name them:
X_trainset, X_testset, y_trainset, y_testset 

The train_test_split will need the parameters: 
X, y, test_size=0.3, and random_state=3. 

The X and y are the arrays required before the split, the test_size represents the ratio of the testing dataset, 
and the random_state ensures that we obtain the same splits.
'''
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

'''
Modeling
We will first create an instance of the DecisionTreeClassifier called drugTree.
Inside of the classifier, specify criterion="entropy" so we can see the information gain of each node.
'''
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree 

#Next, we will fit the data with the training feature matrix X_trainset and training response vector y_trainset
drugTree.fit(X_trainset,y_trainset)

'''
Prediction
Let's make some predictions on the testing dataset and store it into a variable called predTree.
'''
predTree = drugTree.predict(X_testset)

#You can print out predTree and y_testset if you want to visually compare the prediction to the actual values.

print (predTree [0:5])
print (y_testset [0:5])

'''
Evaluation
Next, let's import metrics from sklearn and check the accuracy of our model.
'''
from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))
#Visualization
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
%matplotlib inline 

dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')


