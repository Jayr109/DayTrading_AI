"""
retrain.py

Retrains all strategy models by combining historical CSV bar data with
labeled trades from the feedback loop (trade_log.csv).

Run from project root:
    python src/retrain.py
"""

import os
import sys

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

from features import build_strategy1, build_strategy2, build_strategy3
from train import train_and_save
from trade_logger import get_labeled_trades

RAW_DIR = os.path.join(_PROJECT_ROOT, "data", "raw")

# Strategy 1: single-timeframe (1H), iterated over MNQ1 and MES1
# Strategy 2: multi-timeframe (15M primary + 5M secondary), MNQ1 only
# Strategy 3: multi-timeframe (15M primary + 5M secondary), MES1 only
#
# Each "csvs" entry for multi_tf=False is (symbol, tf_suffix).
# Each "csvs" entry for multi_tf=True  is (symbol, tf_primary, tf_secondary).
STRATEGY_CONFIGS = {
    "strategy1_wick_rejection": {
        "build_fn": build_strategy1,
        "multi_tf": False,
        "csvs": [("MNQ1", "1H"), ("MES1", "1H")],
    },
    "strategy2_orb_retest": {
        "build_fn": build_strategy2,
        "multi_tf": True,
        "csvs": [("MNQ1", "15M", "5M")],
    },
    "strategy3_zone_session": {
        "build_fn": build_strategy3,
        "multi_tf": True,
        "csvs": [("MES1", "15M", "5M")],
    },
}

# Numeric encoding used by build_strategy* (stored in features["strategy"])
_STRATEGY_NUM = {
    "strategy1_wick_rejection": 1.0,
    "strategy2_orb_retest":     2.0,
    "strategy3_zone_session":   3.0,
}

FEEDBACK_WARN_THRESHOLD = 20


def load_bars(symbol: str, tf_suffix: str) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, f"{symbol}_{tf_suffix}.csv")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping.")
        return pd.DataFrame()
    df = pd.read_csv(path, index_col="datetime", parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def build_historical_dataset(strategy_name: str, config: dict):
    """
    Load bars and build features for every symbol/timeframe in the config.
    Returns (X, y) as DataFrames/Series, or (None, None) if no data.
    """
    build_fn = config["build_fn"]
    all_X, all_y = [], []

    for csv_entry in config["csvs"]:
        if config["multi_tf"]:
            symbol, tf_primary, tf_secondary = csv_entry
            df_primary   = load_bars(symbol, tf_primary)
            df_secondary = load_bars(symbol, tf_secondary)
            if df_primary.empty or df_secondary.empty:
                continue
            try:
                X_part, y_part = build_fn(df_primary, df_secondary)
                all_X.append(X_part)
                all_y.append(y_part)
            except Exception as exc:
                print(f"  WARNING: build_features failed for {symbol}: {exc}")
        else:
            symbol, tf_suffix = csv_entry
            bars = load_bars(symbol, tf_suffix)
            if bars.empty:
                continue
            try:
                X_part, y_part = build_fn(bars)
                all_X.append(X_part)
                all_y.append(y_part)
            except Exception as exc:
                print(f"  WARNING: build_features failed for {symbol}_{tf_suffix}: {exc}")

    if not all_X:
        return None, None

    X = pd.concat(all_X).sort_index() if len(all_X) > 1 else all_X[0]
    y = pd.concat(all_y).sort_index() if len(all_y) > 1 else all_y[0]
    return X, y


def extract_feedback(labeled_trades: pd.DataFrame, strategy_name: str):
    """
    Filter labeled_trades to the given strategy and build (X, y) DataFrames.

    "direction" and "strategy" are intentionally kept as feature columns —
    they match what build_strategy* stores in the historical feature matrix.
    The string strategy column is converted to its numeric equivalent (1/2/3).
    """
    METADATA_COLS = {
        "trade_id", "entry_time", "symbol",
        "entry_price", "sl_price", "tp_price", "features_json",
        "exit_time", "exit_price", "label", "pnl_points",
    }

    strat_df = labeled_trades[labeled_trades["strategy"] == strategy_name].copy()
    if strat_df.empty:
        return None, None

    # Convert string strategy to the numeric value used during training
    strat_df["strategy"] = _STRATEGY_NUM.get(strategy_name, 0.0)

    feature_cols = [c for c in strat_df.columns if c not in METADATA_COLS]
    if not feature_cols:
        print(f"  WARNING: No feature columns found in feedback trades for {strategy_name}.")
        return None, None

    X_fb = strat_df[feature_cols].copy()
    y_fb = strat_df["label"].astype(int)
    return X_fb, y_fb


def main():
    print("=" * 60)
    print("Retrain: combining historical bars + feedback trades")
    print("=" * 60)

    try:
        labeled_trades = get_labeled_trades()
    except Exception as exc:
        print(f"WARNING: Could not load labeled trades: {exc}")
        labeled_trades = pd.DataFrame()

    total_feedback = len(labeled_trades)
    print(f"  Total labeled feedback trades: {total_feedback}")
    print()

    for strategy_name, config in STRATEGY_CONFIGS.items():
        print(f"--- {strategy_name} ---")

        X_hist, y_hist = build_historical_dataset(strategy_name, config)
        n_hist = len(y_hist) if y_hist is not None else 0

        X_fb, y_fb = extract_feedback(labeled_trades, strategy_name)
        n_fb = len(y_fb) if y_fb is not None else 0

        if n_fb < FEEDBACK_WARN_THRESHOLD:
            print(
                f"  NOTE: Only {n_fb} feedback samples for {strategy_name}. "
                f"Historical data dominates ({n_hist} samples). "
                f"The feedback loop will strengthen over time."
            )

        if X_hist is None and X_fb is None:
            print(f"  SKIP: No data available for {strategy_name}.")
            print()
            continue

        if X_hist is not None and X_fb is not None:
            if X_hist.shape[1] == X_fb.shape[1]:
                # Align feedback columns to historical order before stacking
                X_fb_aligned = X_fb.reindex(columns=X_hist.columns)
                X_combined = pd.concat(
                    [X_hist.reset_index(drop=True), X_fb_aligned.reset_index(drop=True)]
                ).reset_index(drop=True)
                y_combined = pd.concat(
                    [y_hist.reset_index(drop=True), y_fb.reset_index(drop=True)]
                ).reset_index(drop=True)
            else:
                print(
                    f"  WARNING: Feature dimension mismatch "
                    f"(hist={X_hist.shape[1]}, feedback={X_fb.shape[1]}). "
                    f"Using historical data only."
                )
                X_combined = X_hist.reset_index(drop=True)
                y_combined = y_hist.reset_index(drop=True)
                n_fb = 0
        elif X_hist is not None:
            X_combined = X_hist.reset_index(drop=True)
            y_combined = y_hist.reset_index(drop=True)
        else:
            X_combined = X_fb.reset_index(drop=True)
            y_combined = y_fb.reset_index(drop=True)

        n_combined = len(y_combined)

        try:
            train_and_save(X_combined, y_combined, strategy_name)
            print(
                f"  Trained on {n_combined} total samples "
                f"({n_hist} historical + {n_fb} feedback)."
            )
        except Exception as exc:
            print(f"  ERROR during train_and_save for {strategy_name}: {exc}")

        print()

    print("=" * 60)
    print("Retrain complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
