import numpy as np
from scipy import linalg
from sklearn.datasets import make_regression
from sklearn.linear_model import RidgeCV
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams.update({'font.size': 16})
path_file = os.path.dirname(__file__)
sns.set()
#sns.context('poster')


def get_path(X, y, lbd):
    U, s, Vt = linalg.svd(X, full_matrices=False)
    d = s / (s[:, np.newaxis].T ** 2 + lbd[:, np.newaxis])
    return ((d * (U.T @ y)) @ Vt).T


if __name__ == "__main__":
    n, p = 50, 10
    lbds = np.logspace(-3, 4, 200)
    snr_ = np.array([1, 2, 3, 5, 7, 10])
    for snr in snr_:
        X, y, coeff = make_regression(n, p, coef=True,
                                  random_state=10)
        coeff -= n
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
        plt.savefig(os.path.join(path_file, "..", "prebuilt_images",
                                "path_ridge_coef_"+str(snr)+"_.pdf"))
        plt.show()
        plt.clf()
        plt.close()

        plt.figure()
        plt.plot(lbds, np.mean(loo_cv.cv_values_, axis=0))
        plt.scatter(lbds[np.argmin(np.mean(loo_cv.cv_values_, axis=0))],
                    np.min(np.mean(loo_cv.cv_values_, axis=0)),
                    label="LOO CV min")
        plt.xscale("log")
        plt.xlabel("log regularization")
        plt.yscale('log')
        plt.ylim((10**1, 10**5))
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(path_file, "..", "prebuilt_images",
                                    "path_ridge_loo_"+str(snr)+"_.pdf"))
        plt.show()
        plt.clf()
        plt.close()

        plt.figure(figsize=(7, 9))
        ax1 = plt.subplot(211)
        ax1.vlines(loo_cv.alpha_, np.min(res), np.max(res),
                label=r'optimal $\lambda$',
                color='black',
                linestyles=':')
        for j in range(p):
            ax1.plot(lbds, res[j, :], label=f"coef {j+1}")
            ax1.scatter(max(lbds), coeff[j], label=f"true coef {j+1}")


        ax1.set_xscale("log")
        ax1.set_xlabel("Log Regularization")
        ax1.set_ylabel("Coefficients Values")

        ax2 = plt.subplot(212, sharex=ax1)
        ax2.plot(lbds, np.mean(loo_cv.cv_values_, axis=0))
        ax2.scatter(lbds[np.argmin(np.mean(loo_cv.cv_values_, axis=0))],
                    np.min(np.mean(loo_cv.cv_values_, axis=0)),
                    label="LOO CV min",
                    marker='X', s=100)
        ax2.set_xscale("log")
        ax2.set_xlabel("Log Regularization")
        ax2.set_ylabel("Mean Squared Error (MSE) of LOO-CV")
        ax2.set_yscale('log')
        ax2.set_ylim((10**1, 10**5))

        handles, labels = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
        lgd = plt.legend(handles, labels, bbox_to_anchor=(0.85, 1.5, 0.5, 0.5), ncol=1)
        ttle =plt.suptitle('Leave-One-Out CV')
        plt.savefig(os.path.join(path_file, "..", "prebuilt_images",
                                   "path_ridge_complete_"+str(snr)+"_.pdf"),
                   bbox_extra_artists=(lgd, ttle), bbox_inches='tight')
        plt.show()
        plt.clf()
        plt.close()