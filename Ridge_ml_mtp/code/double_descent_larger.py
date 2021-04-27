import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set()
plt.rcParams.update({'font.size': 16})
seed = 112358
path_file = os.path.dirname(__file__)
save_fig = True


def generate_data(n, p, sigma=.5):
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    beta = np.random.choice([-1., 0., 1.], p)
    beta /= np.linalg.norm(beta, 2)  # normed parameters
    X = rng.multivariate_normal([0], np.eye(1), (n, p)).reshape(n, p)
    noise = rng.multivariate_normal([0], sigma**2 * np.eye(1), (n)).reshape(n)
    y = X @ beta + noise
    return X, y, beta, noise


def new_ridge(n, n_max, p, sigma, ridge=1e-7):
    X, y, _, _ = generate_data(n_max, p, sigma)
    train_reps = []
    test_reps = []
    for i in range(25):
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            train_size=n,
                                                            random_state=i)
        clf = Ridge(alpha=ridge)
        clf.fit(X_train, y_train)
        train_error = mean_squared_error(y_train, clf.predict(X_train))
        train_reps.append(train_error)
        test_error = mean_squared_error(y_test, clf.predict(X_test))
        test_reps.append(test_error)
    return np.mean(train_reps), np.mean(test_reps)


def make_curve(n_samp, n_max, p, sigma, ridge=1e-7):
    train_errors = []
    test_errors = []
    for n in n_samp:
        train, test = new_ridge(n, n_max, p, sigma, ridge)
        train_errors.append(train)
        test_errors.append(test)
    return train_errors, test_errors


def double_descent(n_max, p, sigma):
    ridge_opt = p * sigma ** 2
    n_samples = np.linspace(3, n_max // 2, num=30, dtype=int)
    ridges_low = [1e-6, 1e-3, 1e-2, 1e-1]
    ridges_sup = [1e+1, 1e+2, 1e+3, 1e+6]
    ridges = np.hstack((ridges_low, np.array([1]), ridges_sup))
    all_train = []
    all_test = []
    label = []
    for ridge in ridges:
        train_err, test_err = make_curve(n_samples, n_max, p, sigma,
                                         ridge * ridge_opt)
        all_train.append(train_err)
        all_test.append(test_err)
        label.append(ridge)

    plt.figure()
    for idx in range(len(ridges)):
        if idx <= 3:
            plt.plot(n_samples, all_test[idx],
            label=f"{label[idx]:.1e} ridgeopt",
            linestyle=':', alpha=0.75)
        elif idx == 4:
            plt.plot(n_samples, all_test[idx],
            label=f"{label[idx]:.1e} ridgeopt")
        else:
            plt.plot(n_samples, all_test[idx],
            label=f"{label[idx]:.1e} ridgeopt",
            linestyle='-.', alpha=0.75)

    plt.xlabel('Num Samples')
    plt.ylabel('MSE test')
    plt.title(r'Test Risk for $\ell_2$ Regularized Regression')
    plt.legend()
    plt.ylim((0, 1.6))
    plt.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(path_file, "..", "prebuilt_images",
                                 "double_descent_sup.pdf"))
    plt.show()


if __name__ == '__main__':
    double_descent(1000, 200, .2)
