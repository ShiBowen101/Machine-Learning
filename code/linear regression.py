import numpy as np
import sklearn as sk
import matplotlib
import matplotlib.pyplot as plt
import math

X = np.array([1, 2.2, 3, 3.7, 5, 6])
Y = np.array([3, 4, 5, 6, 7, 8])
plt.plot(X, Y, 'o')

theta0 = 0
theta1 = 0
# theta = [theta1, theta2, theta3]
# prediction = np.dot(theta, X)
# loss=(prediction-Y).T*(prediction-Y)*X.T
X1 = np.dot(X.T, X)
X3 = np.dot(1 / X1, X.T)
X4 = np.dot(X3, Y)
theta = X4
print(theta)
Z = np.dot(theta, X)
plt.plot(X, Z)
Theta = np.array([theta0, theta1])
step = 0.001
for i in range(1, 100000, 1):
    sum_x2 = np.sum((theta1 * X * X), axis=0)
    sum_xy = np.sum((Y * X), axis=0)
    sum_1 = np.sum(np.dot(theta0, X), axis=0)
    descent1 = sum_x2 + sum_1 - sum_xy
    theta1 = theta1 - step * descent1
    sum_theta_x = np.sum(np.dot(theta1, X), axis=0)
    sum_2 = np.sum((theta0 - Y), axis=0)
    descent0 = sum_2 + sum_theta_x
    theta0 = theta0 - step * descent0
    if (abs(descent0) < 0.0001) and (abs(descent1) < 0.0001):
        break
print(theta0, theta1)
Z1 = np.dot(theta1, X) + theta0
plt.plot(X, Z1)

# 简化计算算法

theta_special0 = 0
theta_special1 = 0
step_special = 1
for i in range(1, 6, 1):
    print("special", i, step_special)
    descent_special0 = theta_special1 * X[i - 1] + theta_special0 - Y[i - 1]
    descent_special1 = X[i - 1] * (theta_special1 * X[i - 1] + theta_special0 - Y[i - 1])
    theta_special0 = theta_special0 - descent_special0 * step_special
    theta_special1 = theta_special1 - descent_special1 * step_special
    step_special = step_special / 6
Z2 = X * theta_special1 + theta_special0
plt.plot(X, Z2)
theta_special0 = 0
theta_special1 = 0
step_special = 1
for i in range(1, 6, 1):
    print("special", i, step_special)
    descent_special1 = X[i - 1] * (theta_special1 * X[i - 1] + theta_special0 - Y[i - 1])
    descent_special0 = theta_special1 * X[i - 1] + theta_special0 - Y[i - 1]
    theta_special0 = theta_special0 - descent_special0 * step_special
    theta_special1 = theta_special1 - descent_special1 * step_special
    step_special = step_special / 10
Z3 = X * theta_special1 + theta_special0
plt.plot(X, Z3)
plt.legend(labels=['datas', 'equation', 'descent method', 'small calculate step/6', 'small calculate step/10'])
plt.show()
print("theta0", theta_special0)

# 权重拟合待
