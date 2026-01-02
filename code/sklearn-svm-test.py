import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np
from sklearn.svm import SVC
from sklearn import datasets

stack = datasets.load_iris()  # 返回值类似与字典
data = stack['data']
target = stack['target']
svm_clf_g = SVC(kernel='rbf')
svm_clf_g.fit(data, target)
Y_predict = svm_clf_g.predict(data)
# accuracy
count = 0
for i in range(0, np.size(target), 1):
    if Y_predict[i] == target[i]:
        count = count + 1
accuracy = count / np.size(target)
print("Gaussian accuracy:", accuracy)

svm_clf_l = SVC(kernel='linear')
svm_clf_l.fit(data, target)
Y_predict = svm_clf_l.predict(data)
# accuracy
count = 0
for i in range(0, np.size(target), 1):
    if Y_predict[i] == target[i]:
        count = count + 1
accuracy = count / np.size(target)
print("linear accuracy:", accuracy)

svm_clf_s = SVC(kernel='sigmoid')
svm_clf_s.fit(data, target)
Y_predict = svm_clf_s.predict(data)
# accuracy
count = 0
for i in range(0, np.size(target), 1):
    if Y_predict[i] == target[i]:
        count = count + 1
accuracy = count / np.size(target)
print("Sigmoid accuracy:", accuracy)

svm_clf_p = SVC(kernel='poly')
svm_clf_p.fit(data, target)
Y_predict = svm_clf_p.predict(data)
# accuracy
count = 0
for i in range(0, np.size(target), 1):
    if Y_predict[i] == target[i]:
        count = count + 1
accuracy = count / np.size(target)
print("Poly accuracy:", accuracy)
# sigmoid 发生甚么事了
