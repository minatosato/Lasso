#
# Copyright (c) 2016-2021 Minato Sato
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from sklearn import datasets


def _min_pos(values: np.ndarray) -> float:
    # Only positive finite step lengths are reachable along the current path.
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) == 0:
        return np.inf
    return values.min()


class LarsLasso:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def predict(self, X: np.ndarray):
        y = np.dot(X, self.coef_) + self.intercept_
        return y

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape

        # The LARS derivation assumes centered y and predictors; recover the
        # original-scale intercept after the coefficient path is selected.
        X_offset = X.mean(axis=0)
        y_offset = y.mean()
        X = X - X_offset
        y = y - y_offset
        self.coef_ = np.zeros(p)

        active_set = []
        inactive_set = list(range(p))
        beta = np.zeros(p)
        mu = np.zeros(n)

        change_sign_flag = False

        k = 0
        while k != min(p, n - 1):
            # Correlations between each predictor and the current residual.
            c = np.dot(X.T, y - mu)
            if change_sign_flag:
                # Lasso modification: after removing a zero-crossing coefficient,
                # recompute the direction before adding another variable.
                pass
            else:
                # Add the inactive predictor most correlated with the residual.
                j = inactive_set[np.argmax(np.abs(c[inactive_set]))]
                active_set.append(j)
                inactive_set.remove(j)
            C = np.amax(np.abs(c))

            # Use signed active predictors so every active correlation is +C.
            s = np.sign(c[active_set]).reshape((1, len(active_set)))
            XA = np.copy(X[:, active_set] * s)

            GA = XA.T @ XA
            GA_inv = np.linalg.inv(GA)

            one = np.ones((len(active_set), 1))
            AA = (1. / np.sqrt(one.T @ GA_inv @ one)).flatten()[0]

            # w gives the active coefficient direction and u is the unit
            # equiangular direction that has equal angle to every active column.
            w = AA * GA_inv @ one
            u = XA @ w

            # a_j measures how fast each residual correlation changes along u.
            a = (X.T @ u).ravel()
            d = (s.T * w).ravel()

            if k == p - 1:
                gamma = C / AA
            else:
                # Move until an inactive predictor reaches the active correlation.
                gamma_candidates = np.zeros((len(inactive_set), 2))
                for _j, jj in enumerate(inactive_set):
                    gamma_candidates[_j] = [(C - c[jj]) / (AA - a[jj]), (C + c[jj]) / (AA + a[jj])]
                gamma = _min_pos(gamma_candidates)

            # Lasso-LARS must stop sooner if an active coefficient would change sign.
            gamma_candidates_tilde = - beta[active_set] / d
            gamma_tilde = _min_pos(gamma_candidates_tilde)

            change_sign_flag = False
            if gamma_tilde < gamma:
                gamma = gamma_tilde
                j = active_set[list(gamma_candidates_tilde).index(gamma)]
                change_sign_flag = True

            new_beta = beta[active_set] + gamma * d
            tmp_beta = np.zeros(p)
            tmp_beta[active_set] = new_beta.copy()

            # Along the path, alpha equals max residual correlation divided by n.
            lambda_ = np.max(np.abs(X.T @ (y - X @ tmp_beta))) / n

            if lambda_ < self.alpha:
                prev_lambda_ = np.max(np.abs(X.T @ (y - X @ self.coef_))) / n
                if len(active_set) < 2 and prev_lambda_ < self.alpha:
                    break
                # Interpolate the current segment to stop exactly at the requested alpha.
                modified_gamma = 0 + (gamma - 0) * (self.alpha - prev_lambda_) / (lambda_ - prev_lambda_)
                beta[active_set] += modified_gamma * d
                mu = mu + modified_gamma * u.flatten()
                self.coef_ = beta.copy()
                break

            mu = mu + gamma * u.flatten()
            beta[active_set] = new_beta.copy()
            self.coef_ = beta.copy()

            if change_sign_flag:
                active_set.remove(j)
                inactive_set.append(j)
            k = len(active_set)
        # Convert centered-data coefficients back to the original intercept.
        self.intercept_ = y_offset - X_offset @ self.coef_
        return self


if __name__ == "__main__":
    dataset = datasets.load_diabetes()
    X = dataset.data
    y = dataset.target

    X = (X - X.mean(axis=0, keepdims=True)) / X.std(axis=0, keepdims=True)
    model = LarsLasso(alpha=1)
    model.fit(X, y)

    print(model.intercept_)
    print(model.coef_)
