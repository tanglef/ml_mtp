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

def make_interaction(X):
    n_samples, n_features = X.shape
    Z = np.zeros(shape=(n_samples, int(n_features*(n_features+1)/2)))
    jj = 0
    for j1 in range(n_features):
        for j2 in range(j1, n_features):
            Z[:, jj] += X[:, j1] * X[:, j2]
            jj += 1
    return(Z)

def generate_data_snr(n, p, snr=5):
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    beta = np.random.choice([-1., 0., 1.], p)
    beta /= np.linalg.norm(beta, 2)  # normed parameters
    X = rng.multivariate_normal([0], np.eye(1), (n, p)).reshape(n, p)
    Z = make_interaction(X)
    q = Z.shape[1]
    theta = np.zeros(q)
    # theta = np.random.choice([-1., 0., 0., 0., 1.], q)
    # theta /= np.linalg.norm(theta, 2)  # normed parameters
    sigma = np.sqrt(beta.T@X.T@X@beta)/(np.sqrt(n)*snr)
    noise = rng.multivariate_normal([0], sigma**2 * np.eye(1), (n)).reshape(n)
    y = X @ beta + Z @ theta  + noise
    return X, Z, y, beta, theta, noise


# replicate on one unique sample of size n
def new_ols(n_train, n_test, p_train, p, snr):
    n = n_train + n_test
    X, Z, y, _, _, _ = generate_data_snr(n, p, snr)
    W = np.hstack([X, Z])
    train_reps = []
    test_reps = []
    # print(W.shape)
    for i in range(25):
        W_train, W_test, y_train, y_test = train_test_split(W[:, :p_train], y,
                                                            train_size=n_train,
                                                            random_state=i)
        clf = LinearRegression()
        clf.fit(W_train, y_train)
        train_error = mean_squared_error(y_train, clf.predict(W_train))
        train_reps.append(train_error)
        test_error = mean_squared_error(y_test, clf.predict(W_test))
        test_reps.append(test_error)
    # print(W_train.shape)
    return np.mean(train_reps), np.mean(test_reps)


# execute on various sample size :  n_samp
def make_curve(n_train, n_test, p_train_, p, snr):
    train_errors = []
    test_errors = []
    for p_train in p_train_:
        train, test = new_ols(n_train, n_test, p_train, p, snr)
        train_errors.append(train)
        test_errors.append(test)
    return train_errors, test_errors


def double_descent(n_train, n_test, p, snr):
    q = int(p*(p+1)/2)
    p_train_ = np.linspace(2, 2*p, num=2*p, dtype=int)
    # print(p_train_)
    # return
    train_err, test_err = make_curve(n_train, n_test, p_train_, p, snr)
    all_train = train_err
    all_test = test_err

    plt.figure()
    plt.plot(p_train_, all_test, label='OLS')
    plt.xlabel('n features')
    plt.ylabel('MSE test log-scaled')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(path_file, "..", "prebuilt_images",
                                 "ols_fail_log_features.pdf"))
    plt.show()


if __name__ == '__main__':
    double_descent(100, 100, 100, 5)