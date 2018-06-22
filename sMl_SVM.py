import numpy as np
from sklearn.datasets import load_digits
from sklearn.svm import SVC
#loaing digit images
digit = load_digits()
#only feature data
training_data = digit.data
#only target data
training_target = digit.target
#training data extract from original data
td_original = np.delete(training_data,-1,axis=0)
#training target extract from original data
tt_original = np.delete(training_target,-1)
#now calling support vector machine
clf = SVC()
#Now time for training algo
trained = clf.fit(td_original,tt_original)
#Now time forv predicting data
output = trained.predict(digit.data[-1].reshape(1,64))
print(output)
#plotting that testing image
import matplotlib.pyplot as plt
plt.imshow(digit.images[-1])
plt.show()
