import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
import torch
from torch.utils.data import TensorDataset, DataLoader
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
    X, y, beta = make_regression(n, p, random_state=seed, coef=True)
    X -= np.mean(X, 0)
    X /= np.linalg.norm(X, axis=0)
    X = torch.from_numpy(X).requires_grad_().float()
    y = torch.from_numpy(y).float()
    beta = torch.from_numpy(beta).float()
    return X, y, beta


class LinearReg(nn.Module):
    def __init__(self, p, phi):
        super(LinearReg, self).__init__()
        self.linear = torch.nn.Linear(p, 1)
        self.dropout = nn.Dropout(phi)

    def forward(self, x):
        out = self.dropout(self.linear(x))
        return out


def new_ridge(n, p, phi):
    X, y, _ = generate_data(n, p)
    X = X.detach().numpy()
    y = y.numpy()
    ridge = phi / (1 - phi)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=.3,
                                                        random_state=seed)
    clf = Ridge(alpha=ridge)
    clf.fit(X_train, y_train)
    train_error = mean_squared_error(y_train, clf.predict(X_train))
    test_error = mean_squared_error(y_test, clf.predict(X_test))
    print(f"Ridge test error = {test_error}")
    return train_error, test_error, clf.coef_


def withtorch(n, p, phi, n_epoch):
    X, y, _ = generate_data(n, p)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=.3,
                                                        random_state=seed)
    linearmodel = LinearReg(p, phi)
    optim = torch.optim.SGD(linearmodel.parameters(), lr=2e-4)
    loss = F.mse_loss
    train_ds = TensorDataset(X_train, y_train.requires_grad_())
    train_dl = DataLoader(train_ds, shuffle=True)
    for epoch in range(n_epoch):
        for xb, yb in train_dl:
            optim.zero_grad()
            preds = linearmodel(xb)
            val_loss = loss(preds, yb.view(yb.size(0), -1))
            val_loss.backward(retain_graph=True)
            optim.step()

        if epoch % 100 == 0:
            training_loss = loss(linearmodel(X_train),
                                 y_train.view(y_train.size(0), -1))
            print(f'## Epoch {epoch+1}/{n_epoch}, loss = {training_loss:.3f}')
            linearmodel.eval()
            with torch.no_grad():
                test_pred = linearmodel(X_test)
                val_test = loss(test_pred, y_test.view(y_test.size(0), -1))
            linearmodel.train()
            print(f'~ test loss {val_test:.3f}')
    return val_loss.detach(), val_test, linearmodel.linear.weight[0]


def make_curve(n, p, phi, n_epoch):
    _, _, beta_ridge = new_ridge(n, p, phi)
    _, _, beta_nn = withtorch(n, p, phi, n_epoch)
    return beta_ridge, beta_nn


def plot_coefs(n, p, phi, n_epoch):
    beta_ridge, beta_nn = make_curve(n, p, phi, n_epoch)
    beta_nn = beta_nn.detach().numpy()
    print(beta_nn)
    print(beta_ridge)
    plt.figure()
    plt.plot(beta_ridge, label="ridge")
    plt.plot(beta_nn, label='dropout', marker="*")
    plt.legend()
    plt.tight_layout()
    if save_fig:
        plt.savefig(os.path.join(path_file, "..", "prebuilt_images",
                                 "dropout.pdf"))
    plt.show()


if __name__ == "__main__":
    plot_coefs(500, 20, .2, 5000)
