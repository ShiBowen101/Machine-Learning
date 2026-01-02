import random
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import math

X = np.arange(-10, 10, 1)
data1_x = np.array([5, 6, 7, 8, 9, 10, 11])
data1_y = np.array([5, 5, 5, 5, 5, 5, 5])
data3_x = np.array([-10, -9, -8, -7, -6, -5, -4])
data3_y = np.array([-2, -2, -2, -2, -2, -2, -2])
data2_x = data3_x
data2_y = np.array([4, 4, 4, 4, 4, 4, 4])
i = 0
for letter in data1_y:
    data1_y[i] = letter + random.uniform(-2, 2)
    data2_y[i] = 5 + random.uniform(-2, 2)
    data3_y[i] = -2 + random.uniform(-2, 2)
    i = 1 + i
plt.plot(data1_x, data1_y, 'o')
plt.plot(data2_x, data2_y, 'o')
plt.plot(data3_x, data3_y, 'o')

theta1 = [-1, 1]
theta2 = [1, 1]
theta3 = [2, 2]
while 1:
    sign = 0
    for letter in range(0, data1_x.size - 1, 1):
        if theta1[0] * data1_x[letter] + theta1[1] * data1_y[letter] < 0:
            theta1[0] = theta1[0] + data1_x[letter]
            theta1[1] = theta1[1] + data1_y[letter]
            sign = 1
    if sign == 0:
        break
line1_x = X
line1_y = -theta1[0] / theta1[1] * X
plt.plot(line1_x, line1_y)
while 1:
    sign = 0
    for letter in range(0, data2_x.size - 1, 1):
        if theta2[0] * data2_x[letter] + theta2[1] * data2_y[letter] < 0:
            theta2[0] = theta2[0] + data2_x[letter]
            theta2[1] = theta2[1] + data2_y[letter]
            sign = 1
    if sign == 0:
        break
line2_x = X
line2_y = -theta2[0] / theta2[1] * X
plt.plot(line2_x, line2_y)
while 1:
    sign = 0
    for letter in range(0, data3_x.size - 1, 1):
        if theta3[0] * data3_x[letter] + theta3[1] * data3_y[letter] < 0:
            theta3[0] = theta3[0] + data3_x[letter]
            theta3[1] = theta3[1] + data3_y[letter]
            sign = 1
    if sign == 0:
        break
line3_x = X
line3_y = -theta3[0] / theta3[1] * X
plt.plot(line3_x, line3_y)

print("请输入待测点的坐标X：")
x_p = int(input())
print("请输入待测点的坐标Y：")
y_p = int(input())
plt.plot(x_p, y_p, 'o')
plt.legend(labels=["data1", "data2", "data3", "line1", "line2", "line3", "prediction"])
plt.show()
Value1 = math.exp(x_p * theta1[0] + y_p * theta1[1])
Value2 = math.exp(x_p * theta2[0] + y_p * theta2[1])
Value3 = math.exp(x_p * theta3[0] + y_p * theta3[1])
sum_V = Value1 + Value2 + Value3
Value1 = Value1 / sum_V
Value2 = Value2 / sum_V
Value3 = Value3 / sum_V
V_max = max(Value1, Value2, Value3)
if V_max == Value1:
    print("属于类别1。")
if V_max == Value2:
    print("属于类别2。")
if V_max == Value3:
    print("属于类别3。")
print("P_1:", Value1, theta1)
print("P_2:", Value2, theta2)
print("P_3:", Value3, theta3)
# exp的引入既可以扩大概率差，又可以排除Value<0的情况
