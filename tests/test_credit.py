"""Tests on small made-up data, so they run fast and do not depend on the dataset."""
import numpy as np
import pandas as pd

from src.evaluate import best_threshold, expected_cost
from src.explain import explain_one, logistic_drivers
from src.model import logistic


def toy():
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({"duration": rng.integers(6, 60, n), "credit_amount": rng.integers(500, 15000, n),
                      "housing": rng.choice(["own", "rent", "free"], n)})
    y = ((X["duration"] > 36) | (rng.random(n) < 0.1)).astype(int)
    return X, y


def test_cost_counts_errors_correctly():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1])
    # threshold 0.5: refuse idx 0 and 2 -> approve one bad (cost 5), refuse one good (cost 1) -> 6/4
    assert expected_cost(y, p, 0.5) == 1.5
    assert expected_cost(y, p, 1.01) == 10 / 4   # approve everyone: two bad borrowers approved


def test_best_threshold_beats_approving_everyone():
    X, y = toy()
    p = logistic(X).fit(X, y).predict_proba(X)[:, 1]
    _, cost = best_threshold(y.values, p)
    assert cost < expected_cost(y.values, p, 1.01)


def test_model_learns_the_obvious_driver():
    X, y = toy()
    drivers = logistic_drivers(logistic(X).fit(X, y))
    assert drivers.iloc[0]["feature"] == "duration" and drivers.iloc[0]["effect_on_default"] > 0


def test_explain_one_returns_named_contributions():
    X, y = toy()
    m = logistic(X).fit(X, y)
    out = explain_one(m, X.iloc[[0]])
    assert len(out) > 0 and "effect_on_default" in out.columns
