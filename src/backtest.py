"""
backtest.py

Vectorbt-based backtesting for all three strategies.
Simulates entries based on the same signal logic as features.py, applies
the trained XGBoost model as a trade filter (only take trades where P(win) >= threshold),
and reports performance metrics.

Usage:
    python src/backtest.py --strategy 1 --threshold 0.55
    python src/backtest.py --strategy all --threshold 0.55
    python src/backtest.py --strategy 2 --no-model  # raw strategy, no ML filter
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from collect_data import load_bars
from features import build_strategy1, build_strategy2, build_strategy3

warnings.filterwarnings("ignore")

try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False
    print("WARNING: vectorbt not installed. Install with: pip install vectorbt")
    print("Falling back to manual P&L calculation.\n")

BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# ---------------------------------------------------------------------------
# Manual P&L fallback (if vectorbt not available)
# ---------------------------------------------------------------------------

def manual_backtest(
    trades: pd.DataFrame,
    initial_capital: float = 50_000.0,
) -> dict:
    """
    Simple trade-by-trade P&L simulation.

    trades DataFrame must have columns:
        entry, sl, tp, direction (1=long, -1=short), label (1=win, 0=loss)
    """
    equity = initial_capital
    equity_curve = [equity]
    wins, losses = 0, 0

    for _, row in trades.iterrows():
        risk   = abs(row["entry"] - row["sl"])
        reward = abs(row["tp"] - row["entry"])

        if row["label"] == 1:
            equity += reward
            wins   += 1
        else:
            equity -= risk
            losses += 1

        equity_curve.append(equity)

    total  = wins + losses
    wr     = wins / total if total else 0
    net_pnl = equity - initial_capital
    peak   = initial_capital
    max_dd = 0.0

    for e in equity_curve:
        peak   = max(peak, e)
        max_dd = max(max_dd, peak - e)

    return {
        "total_trades":  total,
        "win_rate":      wr,
        "net_pnl":       net_pnl,
        "max_drawdown":  max_dd,
        "final_equity":  equity,
        "return_pct":    net_pnl / initial_capital * 100,
    }


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

def run_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    strategy_name: str,
    model_name: str,
    use_model: bool = True,
    threshold: float = 0.55,
    initial_capital: float = 50_000.0,
):
    print(f"\n{'='*60}")
    print(f"Backtesting: {strategy_name}")
    print(f"  Samples: {len(X)} | Baseline win rate: {y.mean():.1%}")
    print(f"  ML filter: {'ON (threshold=' + str(threshold) + ')' if use_model else 'OFF'}")
    print(f"{'='*60}")

    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]

    if len(X) == 0:
        print("  No valid samples. Skipping.")
        return

    # Apply ML filter if requested
    selected = pd.Series(True, index=X.index)

    if use_model:
        model_path = MODELS_DIR / f"xgb_{model_name}.pkl"
        feat_path  = MODELS_DIR / f"features_{model_name}.txt"

        if not model_path.exists():
            print(f"  Model not found at {model_path}. Run train.py first.")
            use_model = False
        else:
            model    = joblib.load(model_path)
            features = feat_path.read_text().strip().split("\n")

            # Align features (model may have been trained on a subset)
            available = [f for f in features if f in X.columns]
            proba    = model.predict_proba(X[available].values)[:, 1]
            selected = pd.Series(proba >= threshold, index=X.index)

            filtered_wr = y[selected].mean() if selected.sum() > 0 else 0
            print(f"\n  Trades taken (filtered): {selected.sum()} / {len(X)}")
            print(f"  Filtered win rate: {filtered_wr:.1%}  (baseline: {y.mean():.1%})")

    trades_taken = y[selected]

    if trades_taken.empty:
        print("  No trades passed the filter.")
        return

    # Build a minimal trade frame for P&L
    trade_frame = X[selected].copy()
    trade_frame["label"] = trades_taken

    # Manual P&L
    results = manual_backtest(trade_frame.assign(
        entry=0, sl=trade_frame.get("sl_distance", 10), tp=trade_frame.get("sl_distance", 10) * 3,
    ), initial_capital=initial_capital)

    print(f"\n  Results:")
    print(f"    Total trades:  {results['total_trades']}")
    print(f"    Win rate:      {results['win_rate']:.1%}")
    print(f"    Net P&L:       ${results['net_pnl']:+,.2f}")
    print(f"    Max drawdown:  ${results['max_drawdown']:,.2f}")
    print(f"    Final equity:  ${results['final_equity']:,.2f}")
    print(f"    Return:        {results['return_pct']:+.2f}%")

    return results


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def sweep_thresholds(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    thresholds: list[float] = None,
):
    """
    Find the optimal probability threshold by sweeping values and reporting
    win rate and trade count. Helps choose the right filter level.
    """
    if thresholds is None:
        thresholds = [0.45, 0.50, 0.52, 0.55, 0.57, 0.60, 0.65, 0.70]

    model_path = MODELS_DIR / f"xgb_{model_name}.pkl"
    feat_path  = MODELS_DIR / f"features_{model_name}.txt"

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    model    = joblib.load(model_path)
    features = feat_path.read_text().strip().split("\n")
    available = [f for f in features if f in X.columns]

    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    proba = model.predict_proba(X[available].values)[:, 1]

    print(f"\nThreshold sweep for {model_name}:")
    print(f"  {'Threshold':>10} | {'Trades':>8} | {'Win Rate':>10} | {'vs Baseline':>12}")
    print(f"  {'-'*48}")

    baseline_wr = y.mean()

    for t in thresholds:
        sel = proba >= t
        count = sel.sum()
        wr = y[sel].mean() if count > 0 else 0
        delta = wr - baseline_wr
        print(f"  {t:>10.2f} | {count:>8} | {wr:>9.1%}  | {delta:>+11.1%}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(strategy: str = "all", threshold: float = 0.55, use_model: bool = True):
    print("Loading data...")

    try:
        mnq_1h  = load_bars("MNQ1!", "1H")
        mnq_15m = load_bars("MNQ1!", "15M")
        mnq_5m  = load_bars("MNQ1!", "5M")
        mes_1h  = load_bars("MES1!", "1H")
        mes_15m = load_bars("MES1!", "15M")
        mes_5m  = load_bars("MES1!", "5M")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Run data collection first via Claude + TradingView MCP.")
        return

    if strategy in ("1", "all"):
        print("\nBuilding Strategy 1 features...")
        X1_mnq, y1_mnq = build_strategy1(mnq_1h)
        X1_mes, y1_mes = build_strategy1(mes_1h)
        X1 = pd.concat([X1_mnq, X1_mes]).sort_index()
        y1 = pd.concat([y1_mnq, y1_mes]).sort_index()
        run_backtest(X1, y1, "1H Wick Rejection", "strategy1_wick_rejection",
                     use_model=use_model, threshold=threshold)
        if use_model:
            sweep_thresholds(X1, y1, "strategy1_wick_rejection")

    if strategy in ("2", "all"):
        print("\nBuilding Strategy 2 features...")
        X2, y2 = build_strategy2(mnq_15m, mnq_5m)
        run_backtest(X2, y2, "15M ORB Retest", "strategy2_orb_retest",
                     use_model=use_model, threshold=threshold)
        if use_model:
            sweep_thresholds(X2, y2, "strategy2_orb_retest")

    if strategy in ("3", "all"):
        print("\nBuilding Strategy 3 features...")
        X3, y3 = build_strategy3(mes_15m, mes_5m)
        run_backtest(X3, y3, "7AM Zone + Session Levels", "strategy3_zone_session",
                     use_model=use_model, threshold=threshold)
        if use_model:
            sweep_thresholds(X3, y3, "strategy3_zone_session")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="all", choices=["1", "2", "3", "all"])
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="Minimum model probability to take a trade (default: 0.55)")
    parser.add_argument("--no-model", action="store_true",
                        help="Run raw strategy without ML filter")
    args = parser.parse_args()
    main(args.strategy, args.threshold, use_model=not args.no_model)
