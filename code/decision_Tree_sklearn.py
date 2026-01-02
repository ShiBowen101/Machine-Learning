import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# 离散型变量的例子
sets = datasets.load_digits()
data = sets['data']
target = sets['target']
X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.1, random_state=43)
# test_size为测试机与训练集的比例，random——state为随机种子用于结果复现。
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
prediction = model.predict(X_test)
count = 0
for i in range(0, np.size(Y_test), 1):
    if prediction[i] == Y_test[i]:
        count = count + 1
accuracy = count / np.size(Y_test)
print(prediction)
print(Y_test)
print(accuracy)
# 连续型变量的例子
sets1 = datasets.load_diabetes()
data1 = sets1['data']
target1 = sets1['target']
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(data1, target1, test_size=0.1, random_state=43)
# test_size为测试机与训练集的比例，random——state为随机种子用于结果复现。
model1 = DecisionTreeClassifier()
model1.fit(X_train1, Y_train1)
prediction1 = model1.predict(X_test1)
for i in range(0, np.size(prediction1), 1):
    print(prediction1[i], Y_test1[i])
# 从目前来看效果并不理想
