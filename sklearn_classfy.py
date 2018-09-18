import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

data = h5py.File('/home/dsl/all_check/aichallenger/quan/桃子疮痂病_train.h5','r')
X_train = np.asarray(data['data'])
y_train = np.asarray(data['label'])

data = h5py.File('/home/dsl/all_check/aichallenger/quan/桃子疮痂病_valid.h5','r')
X_test = np.asarray(data['data'])
y_test = np.asarray(data['label'])

cls_fy = KNeighborsClassifier(10)
cls_fy.fit(X_train,y_train)
pred = cls_fy.predict(X_test)
print(cls_fy.score(X_test, y_test))

cls_fy = SVC()
cls_fy.fit(X_train,y_train)
pred = cls_fy.predict(X_test)
print(cls_fy.score(X_test, y_test))
