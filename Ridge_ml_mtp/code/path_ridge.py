import numpy as np
from scipy import linalg
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams.update({'font.size': 16})
sns.set()


def get_path(X, y, lbd):
    U, s, Vt = linalg.svd(X, full_matrices=False)
    d = s / (s[:, np.newaxis].T ** 2 + lbd[:, np.newaxis])
    return ((d * (U.T @ y)) @ Vt).T


if __name__ == "__main__":
    n, p = 30, 10
    lbds = np.logspace(-3, 4, 200)
    X, y, coeff = make_regression(n, p, coef=True, random_state=10)
    coeff -= 30
    y = X @ coeff
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    res = get_path(X, y, lbds)

    plt.figure()
    for j in range(p):
        plt.plot(lbds, res[j, :], label=f"coef {j}")
    plt.xscale("log")
    plt.xlabel("log regularization")
    plt.ylabel("coefficient value")
    plt.show()

