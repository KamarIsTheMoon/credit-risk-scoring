"""Two models: logistic regression (the benchmark banks use, easy to explain) and gradient boosting."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocessor(X):
    num = X.select_dtypes("number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


def logistic(X) -> Pipeline:
    return Pipeline([("prep", preprocessor(X)), ("model", LogisticRegression(max_iter=1000, C=1.0))])


def boosting(X, seed: int = 42) -> Pipeline:
    return Pipeline([("prep", preprocessor(X)),
                     ("model", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=seed))])
