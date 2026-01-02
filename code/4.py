import numpy as np
import sklearn as sk
import numpy as nd
import matplotlib
import matplotlib.pyplot as plt

X = [1, 2, 3, 4, 5, 6]
Y = [1, 4, 5, 7, 8, 12]
print(X.T)
plt.plot(X, Y, 'o')
plt.show()
theta1 = 0
theta2 = 0
theta3 = 0
#theta = [theta1, theta2, theta3]
#prediction = np.dot(theta, X)
# loss=(prediction-Y).T*(prediction-Y)*X.T
theta=np.dot(np.dot(np.dot(X.T,X)**-1,X.T),Y)