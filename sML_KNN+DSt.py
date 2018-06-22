#!/user/bin/python3
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.neighbors  import  KNeighborsClassifier

#loading iris data
iris = load_iris()

#now splitting
train_iris,test_iris,train_target,test_target=train_test_split(iris.data,iris.target,test_size=0.2)

#calling decisison tree classifier
clf=tree.DecisionTreeClassifier()

#  calling KNN  classifier 
knn=KNeighborsClassifier(n_neighbors=5)

# now training  data using dst
traineddst=clf.fit(train_iris,train_target)

# now training  data using knn
trainedknn=knn.fit(train_iris,train_target)

#  test with test_iris using dst
outputdst=traineddst.predict(test_iris)
                       
#  test with test_iris 
outputknn=trainedknn.predict(test_iris)
print(outputdst)
print(outputknn)                          

# actual output 
print(test_target)

# calculating  accuracy score using dst
pctd=accuracy_score(test_target,outputdst)
print(pctd)

# calculating  accuracy score using knn
pctk=accuracy_score(test_target,outputknn)
print(pctk)
'''
#making graph and create comparision
from sklearn.tree import  export_graphviz
out_data=tree.export_graphviz(clf,out_file=tree.dot,
feature_names=iris.feature_names,class_names=iris.target_names,filled=True,rounded=True)
export_graphviz.Source(out_data)
'''
 # exporting  graph  for decisionTree
tree.export_graphviz(clf, out_file="tree.dot", max_depth=7, feature_names=iris.feature_names, class_names=iris.target_names, filled=True,rounded=True)


