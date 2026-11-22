"""Business-aware evaluation.

Approving a borrower who then defaults costs the bank much more than refusing a good one.
The usual cost matrix for this dataset: approving a bad borrower costs 5, refusing a good one costs 1.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

COST_APPROVE_BAD = 5
COST_REFUSE_GOOD = 1


def expected_cost(y_true, p_default, threshold: float) -> float:
    """Average cost per applicant if we refuse everyone with P(default) >= threshold."""
    refuse = p_default >= threshold
    approve_bad = ((~refuse) & (y_true == 1)).sum()
    refuse_good = (refuse & (y_true == 0)).sum()
    return (approve_bad * COST_APPROVE_BAD + refuse_good * COST_REFUSE_GOOD) / len(y_true)


def best_threshold(y_true, p_default, grid=np.linspace(0.05, 0.95, 91)) -> tuple[float, float]:
    costs = [expected_cost(y_true, p_default, t) for t in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])


def summary(y_true, p_default) -> dict:
    t, c = best_threshold(y_true, p_default)
    refuse = p_default >= t
    return {
        "auc": roc_auc_score(y_true, p_default),
        "best_threshold": t,
        "cost_per_applicant": c,
        "cost_if_approve_all": expected_cost(y_true, p_default, 1.01),
        "refusal_rate": refuse.mean(),
        "defaults_caught": (refuse & (y_true == 1)).sum() / max((y_true == 1).sum(), 1),
    }
