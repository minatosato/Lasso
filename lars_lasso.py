#
# Copyright (c) 2016-2021 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


class LarsLasso:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def predict(self, X: np.ndarray):
        y = np.dot(X, self.coef_) + self.intercept_
        return y

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, p = X.shape
        self._beta = []

        self.intercept_ = y.mean()
        y = y - self.intercept_
        self.coef_ = np.zeros(p)

        active_set = []
        inactive_set = list(range(p))
        beta = np.zeros(p)
        mu = np.zeros(n)

        k = 0
        while k != min(p, n - 1):
            c = np.dot(X.T, y - mu)
            # print(np.sign(c[active_set]) * np.sign(beta[active_set]))
            j = inactive_set[np.argmax(np.abs(c[inactive_set]))]
            C = np.amax(np.abs(c))
            active_set.append(j)
            inactive_set.remove(j)
            s = np.sign(c[active_set]).reshape((1, len(active_set)))
            XA = np.copy(X[:, active_set] * s)

            GA = XA.T @ XA
            GA_inv = np.linalg.inv(GA)

            one = np.ones((len(active_set), 1))
            AA = (1. / np.sqrt(one.T @ GA_inv @ one)).flatten()[0]

            w = AA * GA_inv @ one
            u = XA @ w

            a = X.T @ u
            d = s.T * w

            if k == p - 1:
                gamma = C / AA
            else:
                gamma_candidates = np.zeros((len(inactive_set), 2))
                for _j, jj in enumerate(inactive_set):
                    gamma_candidates[_j, 0] = (C - c[jj]) / (AA - a[jj])
                    gamma_candidates[_j, 1] = (C + c[jj]) / (AA + a[jj])
                gamma = gamma_candidates[gamma_candidates > 0].min()

            gamma_candidates_tilde = - beta[active_set] / d.flatten()
            gamma_tilde = gamma_candidates_tilde[gamma_candidates_tilde > 0].min() if len(
                gamma_candidates_tilde[gamma_candidates_tilde > 0]) > 0 else 100000

            flag = False
            if gamma_tilde < gamma:
                gamma = gamma_tilde
                j = active_set[list(gamma_candidates_tilde).index(gamma)]
                flag = True

            new_beta = beta[active_set] + gamma * d.flatten()
            idx = 0 if j != 0 else 1
            tmp_beta = np.zeros(p)
            tmp_beta[active_set] = new_beta.copy()
            lambda_ = np.abs(X[:, active_set[idx]] @ (y - X @ tmp_beta)) * 2 / n
            if lambda_ < self.alpha:
                break

            mu = mu + gamma * u.flatten()
            beta[active_set] = new_beta.copy()
            self.coef_ = beta.copy()

            self._beta.append(self.coef_.copy())
            if flag:
                active_set.remove(j)
                inactive_set.append(j)
            k = len(active_set)
            print(np.round(beta, 3)[6])
        return self


if __name__ == "__main__":
    dataset = datasets.load_diabetes()
    X = dataset.data
    y = dataset.target

    X = (X - X.mean(axis=0, keepdims=True)) / X.std(axis=0, keepdims=True)
    model = LarsLasso(alpha=.0)
    model.fit(X, y)

    print(model.intercept_)
    print(model.coef_)

    # # plot
    # xx = np.sum(np.abs(np.array(model._beta)), axis=1)
    # xx /= xx[-1]
    # plt.plot(xx, np.array(model._beta))
    # (y_min, y_max) = plt.ylim()
    # plt.vlines(xx, y_min, y_max, linestyle='dashed')
    # plt.xlabel(r"$\frac{|\beta_j|}{\max|\beta_j|}$")
    # plt.ylabel('Coefficients')
    # plt.title('LARS Path')
    # plt.axis('tight')
    # plt.show()

    # print(np.linalg.solve(X.T @ X, X.T @ (y - y.mean())))
