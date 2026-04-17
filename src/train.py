"""
train.py

Trains an XGBoost classifier for each strategy (and one combined model).
Evaluates on a walk-forward time-series split to avoid lookahead bias.
Saves trained models to models/.

Usage:
    python src/train.py --strategy 1        # train strategy 1 only
    python src/train.py --strategy 2
    python src/train.py --strategy 3
    python src/train.py --strategy all      # train all + combined (default)
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from collect_data import load_bars
from features import build_strategy1, build_strategy2, build_strategy3

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# XGBoost hyperparameters (conservative defaults for small datasets)
# ---------------------------------------------------------------------------

XGB_PARAMS = {
    "n_estimators":     300,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma":            1.0,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "eval_metric":      "logloss",
    "use_label_encoder": False,
    "random_state":     42,
    "n_jobs":           -1,
}

N_CV_SPLITS = 5   # walk-forward splits


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def prepare_dataset(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Drop NaN rows, align X and y, return arrays."""
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    return X, y


def walk_forward_eval(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Walk-forward cross-validation using TimeSeriesSplit.
    Returns dict of averaged metrics.
    """
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    aucs, accs, win_rates = [], [], []

    X_arr = X.values
    y_arr = y.values

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_arr)):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        if len(np.unique(y_train)) < 2:
            continue  # skip folds with only one class

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        auc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else 0.5
        acc = (preds == y_test).mean()
        wr  = y_test.mean()   # baseline win rate in this fold

        aucs.append(auc)
        accs.append(acc)
        win_rates.append(wr)

        print(f"  Fold {fold+1}: AUC={auc:.3f}  Acc={acc:.3f}  BaselineWR={wr:.3f}")

    return {
        "mean_auc":      np.mean(aucs),
        "mean_accuracy": np.mean(accs),
        "mean_win_rate": np.mean(win_rates),
    }


def train_and_save(
    X: pd.DataFrame,
    y: pd.Series,
    strategy_name: str,
) -> xgb.XGBClassifier:
    """
    Train final model on full dataset and save to models/.
    Returns the trained model.
    """
    X, y = prepare_dataset(X, y)

    if len(X) < 20:
        print(f"  [{strategy_name}] Not enough samples ({len(X)}). Skipping.")
        return None

    print(f"\n{'='*60}")
    print(f"  Strategy: {strategy_name} | Samples: {len(X)} | Win rate: {y.mean():.1%}")
    print(f"{'='*60}")

    print("\nWalk-forward evaluation:")
    metrics = walk_forward_eval(X, y)
    print(f"\nCV Results -> AUC: {metrics['mean_auc']:.3f} | "
          f"Acc: {metrics['mean_accuracy']:.3f} | "
          f"Baseline WR: {metrics['mean_win_rate']:.3f}")

    # Train final model on all data
    print("\nTraining final model on full dataset...")
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X.values, y.values, verbose=False)

    # Feature importance
    importances = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)
    print("\nTop features:")
    print(importances.head(8).to_string())

    # Save model and feature list
    out_path = MODELS_DIR / f"xgb_{strategy_name}.pkl"
    feature_path = MODELS_DIR / f"features_{strategy_name}.txt"
    joblib.dump(model, out_path)
    feature_path.write_text("\n".join(X.columns.tolist()))

    print(f"\nModel saved -> {out_path}")
    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(strategy: str = "all"):
    print("Loading data...")

    # Load bars per instrument
    try:
        mnq_1h  = load_bars("MNQ1!", "1H")
        mnq_15m = load_bars("MNQ1!", "15M")
        mnq_5m  = load_bars("MNQ1!", "5M")
        mes_1h  = load_bars("MES1!", "1H")
        mes_15m = load_bars("MES1!", "15M")
        mes_5m  = load_bars("MES1!", "5M")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Run data collection first. Ask Claude to pull OHLCV data via TradingView MCP.")
        return

    if strategy in ("1", "all"):
        print("\nBuilding Strategy 1 features (1H Wick Rejection — MNQ1 + MES1)...")
        X1_mnq, y1_mnq = build_strategy1(mnq_1h)
        X1_mes, y1_mes = build_strategy1(mes_1h)
        X1 = pd.concat([X1_mnq, X1_mes]).sort_index()
        y1 = pd.concat([y1_mnq, y1_mes]).sort_index()
        train_and_save(X1, y1, "strategy1_wick_rejection")

    if strategy in ("2", "all"):
        print("\nBuilding Strategy 2 features (15M ORB Retest — MNQ1)...")
        X2, y2 = build_strategy2(mnq_15m, mnq_5m)
        train_and_save(X2, y2, "strategy2_orb_retest")

    if strategy in ("3", "all"):
        print("\nBuilding Strategy 3 features (7AM Zone + Session Levels — MES1)...")
        X3, y3 = build_strategy3(mes_15m, mes_5m)
        train_and_save(X3, y3, "strategy3_zone_session")

    if strategy == "all":
        print("\nBuilding combined model (all strategies)...")
        # Stack all features with common columns only
        common_cols = ["direction", "sl_distance", "atr", "hour_cst"]
        dfs = []
        for X, y, s in [
            (X1, y1, 1), (X2, y2, 2), (X3, y3, 3)
        ]:
            available = [c for c in common_cols if c in X.columns]
            sub = X[available].copy()
            sub["strategy"] = s
            sub["label"] = y
            dfs.append(sub)

        combined = pd.concat(dfs).sort_index().dropna()
        Xc = combined.drop(columns=["label"])
        yc = combined["label"]
        train_and_save(Xc, yc, "combined")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        default="all",
        choices=["1", "2", "3", "all"],
        help="Which strategy model to train (default: all)",
    )
    args = parser.parse_args()
    main(args.strategy)
