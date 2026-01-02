import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time

X = np.array([5, 6, 7, 8, 9, 10, 11, 12, 3, 2, 1, 0, -1, -2, -3, -4])
Y = np.array([5, 7, 8, 3, 3, 2, 7, 3, -7, -3, -5, -3, -6, 3 - 7, -2, -4])
plt.figure()
plt.plot(X, Y, 'o')
theta = np.array([-8, 2])
step = 1
for j in range(1, 3, 1):
    for i in range(0, 15, 1):
        value = theta[0] * X[i] + theta[1] * Y[i]
        Y_1 = -theta[0] / theta[1] * X
        X_1 = X
        if value > 0:
            theta[0] = theta[0] + step * X[i]
            theta[1] = theta[1] + step * Y[i]
            plt.plot(X_1, Y_1, )
            print(theta)
        plt.legend(labels=['datas', 'line1', 'line2', 'line3', 'line4', 'line5', 'line6', 'line7', 'line8', 'line9', 'line10'])
plt.show()
# 规范化感知器
# 总结：学习率或者迭代次数的变化均可以改善分类的效果
