#!/user/bin/python3
 
from sklearn import tree

#Features abpout Man,WomenBoy and Girl
#Where 0 means Short, 1 means Long ,2 means soft and 3 means rough
#Attributes are like [weight,height,heir size,skin]
data=[[10,2,1,2],[45,5,1,2],[10,2,0,2],[50,5,0,2],[45,18,1,3],[60,60,1,3],[50,21,0,3],[80,71,0,3]]
output=["Girl","Girl","Boy","Boy","Women","Women","Men","Men"]

#Decision tree algo call
algo = tree.DecisionTreeClassifier()

#train data
trained_algo = algo.fit(data,output)

#Now testing phase
predict = trained_algo.predict([[15,3,1,2]])

#printing output
print(predict)
