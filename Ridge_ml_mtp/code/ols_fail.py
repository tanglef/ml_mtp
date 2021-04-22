import numpy as np
from sklearn.linear_model import LinearRegression
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


# def generate_data(n, p, sigma=.5):
#     np.random.seed(seed)
#     rng = np.random.default_rng(seed)
#     beta = np.random.choice([-1., 0., 1.], p)
#     beta /= np.linalg.norm(beta, 2)  # normed parameters
#     X = rng.multivariate_normal([0], np.eye(1), (n, p)).reshape(n, p)
#     noise = rng.multivariate_normal([0], sigma**2 * np.eye(1), (n)).reshape(n)
#     y = X @ beta + noise
#     return X, y, beta, noise


def generate_data_snr(n, p, snr=5):
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    beta = np.random.choice([-1., 0., 1.], p)
    beta /= np.linalg.norm(beta, 2)  # normed parameters
    X = rng.multivariate_normal([0], np.eye(1), (n, p)).reshape(n, p)
    sigma = np.sqrt(beta.T@X.T@X@beta)/(np.sqrt(n)*snr)
    noise = rng.multivariate_normal([0], sigma**2 * np.eye(1), (n)).reshape(n)
    y = X @ beta + noise
    return X, y, beta, noise


# replicate on one unique sample of size n
def new_ols(n, n_max, p, snr):
    X, y, _, _ = generate_data_snr(n_max, p, snr)
    train_reps = []
    test_reps = []
    for i in range(25):
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            train_size=n,
                                                            random_state=i)
        clf = LinearRegression()
        clf.fit(X_train, y_train)
        train_error = mean_squared_error(y_train, clf.predict(X_train))
        train_reps.append(train_error)
        test_error = mean_squared_error(y_test, clf.predict(X_test))
        test_reps.append(test_error)
    return np.mean(train_reps), np.mean(test_reps)


# execute on various sample size :  n_samp
def make_curve(n_samp, n_max, p, snr):
    train_errors = []
    test_errors = []
    for n in n_samp:
        train, test = new_ols(n, n_max, p, snr)
        train_errors.append(train)
        test_errors.append(test)
    return train_errors, test_errors


def double_descent(n_max, p, snr):
    n_samples = np.linspace(5, n_max // 2, num=159, dtype=int)
    train_err, test_err = make_curve(n_samples, n_max, p, snr)
    all_train = train_err
    all_test = test_err

    plt.figure()
    plt.plot(n_samples, all_test, label='OLS')
    plt.xlabel('n samples')
    plt.ylabel('MSE test log-scaled')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(path_file, "..", "prebuilt_images",
                                 "ols_fail_log.pdf"))
    plt.show()


if __name__ == '__main__':
    double_descent(1000, 200, 8)