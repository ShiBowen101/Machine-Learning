# logistic function:y=1/(1+exp(-X))ory=1/(1+exp(-theta*X))
# dE/dtheta=Y*X/(1+exp(Y*theta.T*X))
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

X = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4])
Y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
theta0 = 0
theta1 = 0
theta = np.array([0, 0])
step = 0.001
descent = np.array([0, 0])
for i in range(1, 1000, 1):
    descent_theta1 = 0
    descent_theta0 = 0
    for j in range(1, 10, 1):
        descent_theta1 = X[j - 1] * (1 / (1 + math.exp(-theta1 * X[j - 1] - theta0)) - Y[j - 1]) + descent_theta1
        descent_theta0 = (1 / (1 + math.exp(-theta0 - theta1 * X[j - 1])) - Y[j - 1]) + descent_theta0
    print("theta:", theta)
    descent[0] = descent_theta0
    descent[1] = descent_theta1
    print("descent:", descent)
    theta = theta - descent * step
    if abs(theta[0] - theta0) < 0.0001 and abs(theta[1] - theta1) < 0.0001:
        break
    theta1 = theta[1]
    theta0 = theta[0]

Z1 = 1 / (1 + np.exp(-theta[1] * X - theta[0]))
for letter in X:
    if Z1[letter] >= 0.5:
        Z1[letter] = 1
    else:
        Z1[letter] = 0
print(Z1)
print(i)
plt.plot(X, Y, 'o')
plt.plot(X, Z1)
plt.show()
