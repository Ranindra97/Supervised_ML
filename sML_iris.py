import numpy
import time
from sklearn.datasets import load_iris
from sklearn import tree
# loading all data
iris = load_iris()

# printing features name
print(iris.feature_names)
#target names
print(iris.target_names) 
x=[0,50,100]
#training target
#Removing exactly one data from each flower
train_data = numpy.delete(iris.data,x,axis = 0)
#Removing exactly one target from each flower
train_target = numpy.delete(iris.target,x)
#target data value
print(train_data)
print(train_target)
#flower data value
print(train_target.size)
#testing data
test_target = iris.target[x]
test_data = iris.data[x]
#decision tree algo
from sklearn import tree
clf = tree.DecisionTreeClassifier()
trained = clf.fit(train_data,train_target)
#testing or prediction of data
output = trained.predict([[7. , 3.2, 4.7, 1.4]])
print(output)
'''
#training data
#features data
#print(iris.data)
#only setosa
#setosa=iris.data[0:50]
#print(setosa)
#target data means flower data
#print(iris.target)
#only setosa data
#s_data = iris.target[0:50]
#print(s_data)
'''
