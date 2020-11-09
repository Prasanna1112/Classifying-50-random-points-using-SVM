import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

np.random.seed(2)

# we create 50 linearly separable points
X = np.r_[np.random.randn(25, 2) - [2, 2], np.random.randn(25, 2) + [2, 2]]
Y = [0] * 25 + [1] * 25

# fit the model
fit_model = svm.SVC(kernel='linear', C=1)
fit_model.fit(X, Y)

# get the separating hyperplane
w = fit_model.coef_[0]
a = -w[0] / w[1]

xx_hor = np.linspace(-5, 5)
yy_vert = a * xx_hor - (fit_model.intercept_[0]) / w[1]

margin = 1 / np.sqrt(np.sum(fit_model.coef_ ** 2))
yy_vert_down = yy_vert - np.sqrt(1 + a ** 2) * margin
yy_vert_up = yy_vert + np.sqrt(1 + a ** 2) * margin


plt.figure(1, figsize=(4, 3))
plt.clf()
plt.plot(xx_hor, yy_vert, "k-")
plt.plot(xx_hor, yy_vert_down, "k-")
plt.plot(xx_hor, yy_vert_up, "k-")
plt.scatter(fit_model.support_vectors_[:, 0], fit_model.support_vectors_[:, 1], s=80,
 facecolors="none", zorder=10, edgecolors="k")


for val, inp in enumerate(Y):
    if Y[val] == 1:
    	plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, marker='X', cmap=plt.cm.Paired, edgecolors="k")
    else:
    	plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, marker='+', cmap=plt.cm.Paired, edgecolors="k")
plt.xlabel("Horizontal Plane")
plt.ylabel("Vertical Plane")
plt.show()