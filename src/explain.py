"""Why does the model accept or refuse a borrower?"""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def logistic_drivers(pipe) -> pd.DataFrame:
    """Coefficients of the logistic model: positive = raises the probability of default."""
    names = pipe.named_steps["prep"].get_feature_names_out()
    coefs = pipe.named_steps["model"].coef_.ravel()
    df = pd.DataFrame({"feature": [n.split("__", 1)[1] for n in names], "effect_on_default": coefs})
    return df.reindex(df["effect_on_default"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def importance(pipe, X, y, seed: int = 42) -> pd.DataFrame:
    """How much the AUC drops when each input is shuffled: works for any model."""
    r = permutation_importance(pipe, X, y, scoring="roc_auc", n_repeats=20, random_state=seed)
    return (pd.DataFrame({"feature": X.columns, "auc_drop": r.importances_mean})
            .sort_values("auc_drop", ascending=False).reset_index(drop=True))


def explain_one(pipe, row: pd.DataFrame, top: int = 4) -> pd.DataFrame:
    """Contribution of each input to one borrower's score (logistic model), versus an average borrower."""
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    x = prep.transform(row)
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    contrib = x.ravel() * model.coef_.ravel()
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    s = pd.Series(contrib, index=names)
    s = s[s != 0]
    return s.reindex(s.abs().sort_values(ascending=False).index).head(top).rename("effect_on_default").to_frame()
