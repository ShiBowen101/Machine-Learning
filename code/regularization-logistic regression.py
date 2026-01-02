import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
import math
from sklearn.linear_model import LogisticRegression

stack = datasets.load_iris()
data = stack['data']
target = stack['target']
# 三分类转化为二分类
count = 0
for i in target:
    if i != 2:
        count = count + 1
# logistic function  F=1/(1+exp(-f(x)))
# 根据极大似然估计，优化函数为：F=sum(-y(i)*w.T*x(i)+In(1+exp(w.T*x(i))))
# 对w求导得：F‘=sum(-y(i)*x(i)+exp(w.T*x(i)/(1+exp(w.T*x(i)))
theta = np.array([[0.1, 0.1, 0.1, 0.1, 0.1]]).T
X = np.array([[1, 1, 1, 1, 1]]).T
I = np.array([[1, 1, 1, 1, 1]]).T
# 注意：向量没有转置
y = 0
step = 0.001
# 迭代次数 N
sign = 0
decent = np.array([[0, 0, 0, 0, 0]]).T
for j in range(0, 200, 1):
    temp = theta
    for i in range(0, count, 1):
        X[0] = data[i, 0]
        X[1] = data[i, 1]
        X[2] = data[i, 2]
        X[3] = data[i, 3]
        y = target[i]
        decent = (y * X - X.dot(np.exp(theta.T.dot(X)) / (1 + np.exp(theta.T.dot(X))))) + decent
    theta = theta + step * decent
    sign_count = 0

    for i in range(0, np.size(theta), 1):
        if abs(theta[i][0] - temp[i][0]) < 0.001:
            sign_count = sign_count + 1
            break
    Re = theta[3][0] / theta[1][0]
    if sign_count == np.size(theta):
        sign = 1
    if sign == 1:
        print(j)
        break
prediction = np.zeros((1, count))
P_0 = 0
P_1 = 0
for i in range(0, count, 1):
    X[0] = data[i, 0]
    X[1] = data[i, 1]
    X[2] = data[i, 2]
    X[3] = data[i, 3]
    P_0 = int(1 / np.exp(theta.T.dot(X)))
    p_1 = 1 - P_0
    if P_0 > P_1:
        prediction[0, i] = 0
    else:
        prediction[0, i] = 1
count_y = 0
prediction = prediction[0]
for i in range(0, count, 1):
    if prediction[i] == target[i]:
        count_y = count_y + 1
accuracy1 = count_y / count
print("right number:", count_y)
print("accuracy1:", accuracy1)
print(prediction)
print(target)
model = LogisticRegression(max_iter=1000)
model.fit(data, target)
prediction_sklearn_sigmoid = model.predict(data)
count_y = 0
for i in range(0, count, 1):
    if prediction_sklearn_sigmoid[i] == target[i]:
        count_y = count_y + 1
accuracy2 = count_y / count
print("right number:", count_y)
print("accuracy2:", accuracy2)
print(target)
print(prediction_sklearn_sigmoid)
# 完结，待总结
# 1.该模型在经过一定轮次的训练后可以将准确率达到100%
# 2.但是存在的问题可能需要引入正则化来阻止参数的膨胀。
# 3.通过Re参数可以得到在训练达到一定程度之后参数间的比例趋于稳定，但是参数的模值在不断增大导致计算数值溢出。
# 4.在训练中注意：np.sum()函数使用向量标量化的计算函数。
# 5.注意矩阵和向量的区别以及1*1矩阵和标量的区别。
# 6.此后有机会进行多分类的改造。
