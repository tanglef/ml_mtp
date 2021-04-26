import numpy as np
from scipy import linalg
from sklearn.datasets import make_regression
from sklearn.linear_model import RidgeCV
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams.update({'font.size': 16})
path_file = os.path.dirname(__file__)
sns.set()


def get_path(X, y, lbd):
    U, s, Vt = linalg.svd(X, full_matrices=False)
    d = s / (s[:, np.newaxis].T ** 2 + lbd[:, np.newaxis])
    return ((d * (U.T @ y)) @ Vt).T


if __name__ == "__main__":
    n, p = 30, 10
    lbds = np.logspace(-3, 4, 200)
    X, y, coeff = make_regression(n, p, coef=True,
                                  random_state=10)
    coeff -= 30
    snr = 5
    sigma = np.sqrt(coeff.T@X.T@X@coeff)/(np.sqrt(n)*snr)
    noise = np.random.normal(scale=sigma, size=n)
    y = X @ coeff + noise
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    y -= np.mean(y)
    res = get_path(X, y, lbds)
    loo_cv = RidgeCV(alphas=lbds,
                     fit_intercept=False,
                     cv=None,
                     store_cv_values=True).fit(X, y)

    plt.figure()
    plt.vlines(loo_cv.alpha_, np.min(res), np.max(res),
               label=r'optimal $\lambda$',
               color='black',
               linestyles=':')
    for j in range(p):
        plt.plot(lbds, res[j, :], label=f"coef {j+1}")
        plt.scatter(max(lbds), coeff[j], label=f"true coef {j+1}")


    plt.xscale("log")
    plt.xlabel("log regularization")
    plt.ylabel("coefficient value")
    plt.legend(ncol=2, bbox_to_anchor=(1.21, 0.4, 0.5, 0.5))
    plt.show()
    plt.savefig(os.path.join(path_file, "..", "prebuilt_images",
                                 "path_ridge_coef.pdf"))
    plt.clf

    plt.figure()
    plt.plot(lbds, np.mean(loo_cv.cv_values_, axis=0))
    plt.scatter(lbds[np.argmin(np.mean(loo_cv.cv_values_, axis=0))],
                 np.min(np.mean(loo_cv.cv_values_, axis=0)),
                 label="LOO CV min")
    plt.xscale("log")
    plt.xlabel("log regularization")
    plt.yscale('log')
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(path_file, "..", "prebuilt_images",
                                 "path_ridge_loo.pdf"))
