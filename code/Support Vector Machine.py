import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np
from sklearn.svm import SVC

# 数据实现部分
data1 = pd.read_excel('data3.0.xlsx')
data = data1.loc[:, '密度':'好瓜']
X_y = np.ones([17])
Y_y = np.ones([17])
X_n = np.ones([17])
Y_n = np.ones([17])
i = 0
index1 = 0
index2 = 0
for letter in data.loc[:, '密度']:
    if data.loc[i, '好瓜'] == '是':
        X_y[index1] = letter
        index1 = index1 + 1
    else:
        X_n[index2] = letter
        index2 = index2 + 1
    i = i + 1
i = 0
index1 = 0
index2 = 0
for letter in data.loc[:, '含糖量']:
    if data.loc[i, '好瓜'] == '是':
        Y_y[index1] = letter
        index1 = index1 + 1
    else:
        Y_n[index2] = letter
        index2 = index2 + 1
    i = i + 1
X_y = X_y[0:index1 - 1]
Y_y = Y_y[0:index1 - 1]
X_n = X_n[0:index2 - 1]
Y_n = Y_n[0:index2 - 1]
i = 0
for letter in data.loc[:, '好瓜']:
    if letter == '是':
        data.loc[i, '好瓜'] = 1
    else:
        data.loc[i, '好瓜'] = -1
    i = i + 1
X = np.zeros((17, 2))
Y = np.zeros((17, 1))

# 数据分割
for count in range(0, 17, 1):
    X[count, 0] = data.loc[count, '密度']
    X[count, 1] = data.loc[count, '含糖量']
    Y[count, 0] = data.loc[count, '好瓜']
Y = Y.flatten()  # !!!!!!!!!!!!!!
print(Y)
# plt.plot(X_y, Y_y, '.')
# plt.plot(X_n, Y_n, '.')
# plt.show()
# 学完运筹学后再回来
# representer theorem
svm_clf_l = SVC(kernel='linear')
svm_clf_l.fit(X, Y)
Y_predict = svm_clf_l.predict(X)
# print("actually:", Y)
# print("prediction:", Y_predict)
# 准确率的计算
count = 0
for i in range(0, 17, 1):
    if Y[i] == Y_predict[i]:
        count = count + 1
accuracy = count / 17
print(Y_predict)
print('accuracy linear:', accuracy)

svm_clf_g = SVC(kernel='rbf')
svm_clf_g.fit(X, Y)
Y_predict = svm_clf_g.predict(X)
count = 0
for i in range(0, 17, 1):
    if Y[i] == Y_predict[i]:
        count = count + 1
accuracy = count / 17
print(Y_predict)
print('accuracy Gaussian:', accuracy)

svm_clf_s = SVC(kernel='sigmoid')
svm_clf_s.fit(X, Y)
Y_predict = svm_clf_s.predict(X)
print(Y_predict)
count = 0
for i in range(0, 17, 1):
    if Y[i] == Y_predict[i]:
        count = count + 1
accuracy = count / 17
print('accuracy Sigmoid:', accuracy)

svm_clf_p = SVC(kernel='poly')
svm_clf_p.fit(X, Y)
Y_predict = svm_clf_p.predict(X)
print(Y_predict)
count = 0
for i in range(0, 17, 1):
    if Y[i] == Y_predict[i]:
        count = count + 1
accuracy = count / 17
print('accuracy Poly:', accuracy)
print(svm_clf_g.decision_function(X))  # 描述对应好坏瓜的映射
# 总结：可以根据几种不同的核函数映射方式选择最优
# 暂时完结


