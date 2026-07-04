import unittest

import numpy as np
from sklearn.linear_model import Lasso

from coordinate_descent_lasso import CoordinateDescentLasso
from lars_lasso import LarsLasso


def make_problem():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    X = X * np.array([1.0, 2.0, 0.5, 3.0, 1.5, 0.8])
    X = X + np.array([2.0, -1.0, 0.5, 3.0, 0.0, -2.0])
    coef = np.array([1.5, 0.0, -2.0, 0.0, 0.5, 0.0])
    y = X @ coef + rng.normal(scale=0.1, size=X.shape[0]) + 3.0
    return X, y


class LassoTest(unittest.TestCase):
    def test_coordinate_descent_matches_sklearn(self):
        X, y = make_problem()
        alpha = 0.1

        expected = Lasso(alpha=alpha, fit_intercept=True, max_iter=100000, tol=1e-12).fit(X, y)
        actual = CoordinateDescentLasso(alpha=alpha, max_iter=10000).fit(X, y)

        np.testing.assert_allclose(actual.coef_, expected.coef_, rtol=1e-9, atol=1e-9)
        self.assertAlmostEqual(actual.intercept_, expected.intercept_, places=9)

    def test_lars_matches_sklearn(self):
        X, y = make_problem()
        alpha = 0.1

        expected = Lasso(alpha=alpha, fit_intercept=True, max_iter=100000, tol=1e-12).fit(X, y)
        actual = LarsLasso(alpha=alpha).fit(X, y)

        np.testing.assert_allclose(actual.coef_, expected.coef_, rtol=1e-9, atol=1e-9)
        self.assertAlmostEqual(actual.intercept_, expected.intercept_, places=9)

    def test_lars_uses_zero_coefficients_when_alpha_is_large(self):
        X, y = make_problem()
        X_centered = X - X.mean(axis=0)
        y_centered = y - y.mean()
        alpha = np.max(np.abs(X_centered.T @ y_centered)) / X.shape[0] + 1.0

        actual = LarsLasso(alpha=alpha).fit(X, y)

        np.testing.assert_allclose(actual.coef_, np.zeros(X.shape[1]), rtol=0.0, atol=1e-12)
        self.assertAlmostEqual(actual.intercept_, y.mean(), places=12)


if __name__ == "__main__":
    unittest.main()
