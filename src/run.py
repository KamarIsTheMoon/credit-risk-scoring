"""Run the whole analysis.

    python -m src.run

Uses 5-fold cross-validation so every borrower is scored by a model that never saw them.
Writes output/results.md and charts to output/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from .data import load, split_xy
from .evaluate import summary
from .explain import explain_one, importance, logistic_drivers
from .model import boosting, logistic

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
SEED = 42


def main() -> dict:
    df = load()
    X, y = split_xy(df)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models = {"Logistic regression": logistic(X), "Gradient boosting": boosting(X, SEED)}
    probs = {name: cross_val_predict(m, X, y, cv=cv, method="predict_proba")[:, 1] for name, m in models.items()}
    results = {name: summary(y.values, p) for name, p in probs.items()}

    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, p in probs.items():
        RocCurveDisplay.from_predictions(y, p, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_title("ROC curve (5-fold cross-validation)")
    fig.tight_layout(); fig.savefig(OUT / "roc_curve.png", dpi=150); plt.close(fig)

    # Explainability on a held-out split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    log_model = logistic(X).fit(Xtr, ytr)
    gb_model = boosting(X, SEED).fit(Xtr, ytr)
    drivers = logistic_drivers(log_model)
    imp = importance(gb_model, Xte, yte, SEED)

    fig, ax = plt.subplots(figsize=(6, 4))
    imp.iloc[::-1].plot.barh(x="feature", y="auc_drop", ax=ax, legend=False, color="#1F3864")
    ax.set_xlabel("Drop in AUC when the input is shuffled"); ax.set_ylabel("")
    ax.set_title("What drives the gradient boosting model")
    fig.tight_layout(); fig.savefig(OUT / "feature_importance.png", dpi=150); plt.close(fig)

    p_test = log_model.predict_proba(Xte)[:, 1]
    riskiest = Xte.index[int(np.argmax(p_test))]
    example = explain_one(log_model, Xte.loc[[riskiest]])

    # Fairness check: approval rate by sex (sex is NOT used by the model)
    t = results["Logistic regression"]["best_threshold"]
    approve = pd.Series(probs["Logistic regression"] < t, index=X.index)
    fairness = approve.groupby(df["sex"]).mean()

    lines = ["# Results", "", f"Data: {len(df)} borrowers, {y.mean():.0%} defaulted. 5-fold cross-validation.", "",
             "| Model | AUC | Best refusal threshold | Refusal rate | Defaults caught | Cost per applicant | Cost if we approve everyone |",
             "|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | {r['auc']:.3f} | P(default) >= {r['best_threshold']:.2f} | {r['refusal_rate']:.0%} | {r['defaults_caught']:.0%} | {r['cost_per_applicant']:.2f} | {r['cost_if_approve_all']:.2f} |")
    lines += ["", "## Strongest drivers in the logistic model (positive = more risk)", "", drivers.head(8).round(3).to_markdown(index=False),
              "", "## Input importance in the gradient boosting model", "", imp.round(4).to_markdown(index=False),
              "", f"## Example: why the riskiest test applicant (P(default) = {p_test.max():.0%}) is refused", "", example.round(3).to_markdown(),
              "", "## Fairness check: approval rate by sex (not an input of the model)", "", fairness.map('{:.0%}'.format).to_frame('approval_rate').to_markdown()]
    md = "\n".join(lines) + "\n"
    (OUT / "results.md").write_text(md, encoding="utf-8")
    print(md)
    return results


if __name__ == "__main__":
    main()
