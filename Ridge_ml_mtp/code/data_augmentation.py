import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set()
plt.rcParams.update({'font.size': 16})
seed = 112358
path_file = os.path.dirname(__file__)
save_fig = True


def generate_data(n, p):
    np.random.seed(seed)
    x1 = np.random.uniform(-2, 2, n)
    x2 = x1 + np.random.normal(0, scale=.5, size=x1.size)
    X = np.vstack((x1, x2)).T
    beta = np.array([-2, 2])
    y = X @ beta
    return X, y, beta


def new_ridge(n, p, pen):
    X, y, _ = generate_data(n, p)
    clf = Ridge(alpha=pen, tol=1e-10, solver='lsqr')
    clf.fit(X, y)
    return clf, X, y


def augmented_data(n, p, m, pen):
    X, y, _ = generate_data(n, p)
    np.random.seed(seed+15)
    rng = np.random.default_rng(seed)
    augmented = []
    n = X.shape[0]
    for i in range(n):
        for _ in range(m):
            new = X[i, :] + rng.multivariate_normal([0], pen / n * np.eye(1), (1, p)).reshape(1, p)
            new = new.reshape(-1, p)
            augmented.append(new)
            X = np.vstack((X, new))
        y = np.hstack((y, np.repeat(y[i], m)))

    clf = LinearRegression()
    clf.fit(X, y)
    return clf, X, y, augmented


def plot_results(n, p, m, pen):
    if p != 2:
        p = 2
    ridge, Xr, _ = new_ridge(n, p, pen)
    ols, _, _, augmented = augmented_data(n, p, m, pen)
    augmented = np.concatenate(augmented, axis=0)

    plt.figure()
    plt.plot(augmented[:, 0], augmented[:, 1], "o", label="augmented data",
             ms=2.5)
    plt.plot(Xr[:, 0], Xr[:, 1], "o", label="original data")
    plt.xlabel(r'$X_1$')
    plt.ylabel(r'$X_2$')
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)

    coef_ridges = ridge.coef_
    coef_ols = ols.coef_
    plt.figure()
    plt.plot(coef_ridges, label="Ridge")
    plt.plot(coef_ols, label="augmented OLS")
    plt.ylabel("Signal value")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    n, p, m, pen = 200, 2, 20, .3
    plot_results(n, p, m, pen)
