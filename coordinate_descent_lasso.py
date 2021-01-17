#
# Copyright (c) 2016-2021 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from sklearn import datasets


class CoordinateDescentLasso:
    def __init__(self, alpha: float = 1.0, max_iter: int = 1000, fit_intercept: bool = True) -> None:
        self.alpha: float = alpha  # 正則化項の係数
        self.max_iter: int = max_iter  # 繰り返しの回数
        self.fit_intercept: bool = fit_intercept  # 切片(i.e., \beta_0)を用いるか
        self.coef_ = None  # 回帰係数(i.e., \beta)保存用変数
        self.intercept_ = None  # 切片保存用変数

    def _soft_thresholding_operator(self, x: float, lambda_: float) -> float:
        if x > 0.0 and lambda_ < abs(x):
            return x - lambda_
        elif x < 0.0 and lambda_ < abs(x):
            return x + lambda_
        else:
            return 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.fit_intercept:
            X = np.column_stack((np.ones(len(X)), X))

        beta = np.zeros(X.shape[1])
        if self.fit_intercept:
            beta[0] = np.sum(y - np.dot(X[:, 1:], beta[1:])) / (X.shape[0])

        for iteration in range(self.max_iter):
            start = 1 if self.fit_intercept else 0
            for j in range(start, len(beta)):
                tmp_beta = beta.copy()
                tmp_beta[j] = 0.0
                r_j = y - np.dot(X, tmp_beta)
                arg1 = np.dot(X[:, j], r_j)
                arg2 = self.alpha * X.shape[0]

                beta[j] = self._soft_thresholding_operator(arg1, arg2) / (X[:, j]**2).sum()

                if self.fit_intercept:
                    beta[0] = np.sum(y - np.dot(X[:, 1:], beta[1:])) / (X.shape[0])

        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.coef_ = beta

        return self

    def predict(self, X: np.ndarray):
        y = np.dot(X, self.coef_)
        if self.fit_intercept:
            y += self.intercept_ * np.ones(len(y))
        return y


if __name__ == "__main__":
    dataset = datasets.load_boston()
    X = dataset.data
    y = dataset.target

    X = (X - X.mean(axis=0, keepdims=True)) / X.std(axis=0, keepdims=True)

    model = CoordinateDescentLasso(alpha=1 / 2)
    model.fit(X, y)

    print(model.intercept_)
    print(model.coef_)
