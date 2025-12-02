from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb


def load_xgb_model(model_path: str | Path) -> xgb.Booster:
    """
    Load an XGBoost model from a joblib-saved file.

    Supports:
      - Plain XGBClassifier/XGBRegressor saved via joblib.dump(model, path)
      - Wrapped dicts like {"model": clf, ...} saved via joblib.dump(...)
    """
    model_path = Path(model_path)
    obj = joblib.load(model_path)

    # unwrap dict-style saves: {"model": clf, ...}
    if isinstance(obj, dict):
        clf = obj.get("model")
        if clf is None:
            raise ValueError(
                f"Loaded dict from {model_path} but it has no 'model' key."
            )
    else:
        clf = obj

    if not hasattr(clf, "get_booster"):
        raise TypeError(
            f"Object loaded from {model_path} does not look like an XGBoost "
            f"estimator (missing get_booster). Type: {type(clf)}"
        )

    booster: xgb.Booster = clf.get_booster()
    return booster


def feature_importance_df(
    booster: xgb.Booster,
    importance_type: str = "gain",
    feature_names: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Return a DataFrame of feature importances.

    importance_type: 'weight', 'gain', 'cover', etc.
    """
    score = booster.get_score(importance_type=importance_type)
    if not score:
        raise ValueError("No feature importance found in booster")

    # XGBoost uses 'f0', 'f1' etc. by default
    items = sorted(score.items(), key=lambda kv: kv[1], reverse=True)
    features, values = zip(*items)

    df = pd.DataFrame({"feature": features, "importance": values})

    if feature_names:
        # Map f0, f1 -> actual names if provided
        mapping = {f"f{i}": name for i, name in enumerate(feature_names)}
        df["feature"] = df["feature"].map(lambda f: mapping.get(f, f))

    return df


def plot_feature_importance(
    model_path: str | Path,
    out_dir: str | Path,
    title: Optional[str] = None,
    importance_type: str = "gain",
    feature_names: Optional[list[str]] = None,
    top_n: int = 20,
) -> Path:
    """
    Plot feature importance for a trained XGBoost model.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    booster = load_xgb_model(model_path)
    df = feature_importance_df(
        booster, importance_type=importance_type, feature_names=feature_names
    )

    df = df.sort_values("importance", ascending=True).tail(top_n)

    plt.figure(figsize=(8, max(4, 0.4 * len(df))))
    plt.barh(df["feature"], df["importance"])
    plt.xlabel(f"Importance ({importance_type})")
    plt.title(title or f"Feature Importance – {model_path.name}")
    plt.tight_layout()

    out_path = out_dir / f"{model_path.stem}_feature_importance.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path
