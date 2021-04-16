import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    X, y, beta = make_regression(n, p, random_state=seed, coef=True, noise=1)
    X_ = np.copy(X)
    X -= np.mean(X, 0)
    X /= np.linalg.norm(X, axis=0)
    X = torch.from_numpy(X).requires_grad_().float()
    y = torch.from_numpy(y).float()
    beta = torch.from_numpy(beta).float()
    return X, y, beta, X_


class LinearReg(nn.Module):
    def __init__(self, p, phi):
        super(LinearReg, self).__init__()
        self.linear = torch.nn.Linear(p, 1)
        self.d = nn.Dropout(phi)

    def forward(self, x):
        out = self.d(x)
        out = self.linear(out)
        return out


def new_ridge(n, p, phi):
    X, y, _, _ = generate_data(n, p)
    X = X.detach().numpy()
    y = y.numpy()
    ridge = phi / (1 - phi)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=.3,
                                                        random_state=seed)
    clf = Ridge(alpha=ridge, tol=1e-10, solver='lsqr')
    clf.fit(X_train, y_train)
    train_error = mean_squared_error(y_train, clf.predict(X_train))
    test_error = mean_squared_error(y_test, clf.predict(X_test))
    print(f"Ridge test error = {test_error}")
    return train_error, test_error, clf


def withtorch(n, p, phi, n_epoch, lr):
    X, y, _, _ = generate_data(n, p)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=.3,
                                                        random_state=seed)
    linearmodel = LinearReg(p, phi)
    optim = torch.optim.SGD(linearmodel.parameters(), lr=lr)
    loss = F.mse_loss
    prev_train_loss = 1e8
    for epoch in range(n_epoch):
        optim.zero_grad()
        preds = linearmodel(X_train)
        val_loss = loss(preds, y_train.view(y_train.size(0), -1))
        val_loss.backward(retain_graph=True)
        optim.step()

        if epoch % 1000 == 0:
            linearmodel.eval()
            with torch.no_grad():
                training_loss = loss(linearmodel(X_train),
                                     y_train.view(y_train.size(0), -1))
                print(f'## Epoch {epoch+1}/{n_epoch}, loss = {training_loss:.3f}')
                test_pred = linearmodel(X_test)
                val_test = loss(test_pred, y_test.view(y_test.size(0), -1))
            linearmodel.train()
            print(f'~ test loss {val_test:.3f}')
            if training_loss > prev_train_loss:
                break
            else:
                prev_train_loss = training_loss
    return val_loss.detach(), val_test, linearmodel


def make_curve(n, p, phi, n_epoch, lr):
    _, _, model_ridge = new_ridge(n, p, phi)
    _, _, model_nn = withtorch(n, p, phi, n_epoch, lr)
    return model_ridge, model_nn


def plot_coefs(n, p, phi, n_epoch, lr):
    ridge, neural = make_curve(n, p, phi, n_epoch, lr)
    plt.figure()
    plt.plot(ridge.coef_, label="ridge")
    plt.plot(neural.linear.weight[0].detach().numpy(),
             marker="*", label="dropout")
    plt.legend()
    plt.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(path_file, "..", "prebuilt_images",
                                 "dropout.pdf"))
    plt.show()
    return ridge, nn


if __name__ == "__main__":
    n, p, phi, n_epoch, lr = 80, 30, .5, 42000, 1e-3
    ridge, neural = plot_coefs(n, p, phi, n_epoch, lr)
