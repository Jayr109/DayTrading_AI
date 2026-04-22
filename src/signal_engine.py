"""
signal_engine.py

Paper trading signal generator for the DayTrading AI system.
Runs at the close of each 1H bar, scores the bar with the trained XGBoost model,
and emits actionable trade signals. Also checks open trades for TP/SL outcomes.

Usage:
    python src/signal_engine.py                    # run once for MNQ1 (default)
    python src/signal_engine.py --symbol MES1      # run for MES1
    python src/signal_engine.py --threshold 0.60   # custom threshold
    python src/signal_engine.py --check-only       # only check pending trades, no new signal
"""

import os
import sys
import argparse
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Path setup -- allows running as `python src/signal_engine.py` from project root
# ---------------------------------------------------------------------------

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import json
import numpy as np
import pandas as pd
import yfinance as yf
import joblib

from features import build_strategy1
from trade_logger import log_trade, close_trade, get_pending_trades

SIGNAL_FILE = os.path.join(_PROJECT_ROOT, "data", "pending_signal.json")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRATEGY_NAME = "strategy1_wick_rejection"

SYMBOL_TO_YF = {
    "MNQ1": "NQ=F",
    "MES1": "ES=F",
}

MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "xgb_strategy1_wick_rejection.pkl")
FEATURES_PATH = os.path.join(_PROJECT_ROOT, "models", "features_strategy1_wick_rejection.txt")

DEFAULT_THRESHOLD = 0.55
DEFAULT_SYMBOL = "MNQ1"

# ATR fallback multipliers
SL_ATR_MULT = 1.0
TP_ATR_MULT = 3.0


# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------

def fetch_bars(yf_ticker: str, n_bars: int = 50) -> pd.DataFrame:
    """
    Pull the last ~50 1H bars from yfinance using period='5d'.
    No datetime.now() -- Yahoo picks the window.
    Returns UTC-indexed DataFrame with lowercase OHLCV columns.
    """
    raw = yf.download(
        tickers=yf_ticker,
        period="5d",
        interval="1h",
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {yf_ticker}")

    # Flatten multi-level columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.columns = [c.lower() for c in raw.columns]
    raw.index.name = "datetime"

    if raw.index.tz is None:
        raw.index = raw.index.tz_localize("UTC")
    else:
        raw.index = raw.index.tz_convert("UTC")

    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in raw.columns]
    df = raw[keep]

    # Keep last n_bars rows
    if len(df) > n_bars:
        df = df.iloc[-n_bars:]

    return df


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    """Load the trained XGBoost model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def load_feature_list() -> list:
    """Load the expected feature column order from the saved txt file."""
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Feature list not found: {FEATURES_PATH}")
    with open(FEATURES_PATH, "r") as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------------
# ATR helper
# ---------------------------------------------------------------------------

def _compute_atr(df: pd.DataFrame, window: int = 14) -> float:
    """Simple ATR estimate from bar ranges over last `window` bars."""
    ranges = df["high"] - df["low"]
    atr = float(ranges.tail(window).mean())
    return atr if not np.isnan(atr) else 1.0


# ---------------------------------------------------------------------------
# Signal scoring
# ---------------------------------------------------------------------------

def score_bar(df_bars: pd.DataFrame, model, feature_cols: list) -> dict:
    """
    Build features from the bar data, score the LAST row with the model.

    Returns a dict with keys:
        prob, direction, entry_price, sl_price, tp_price, rr_ratio,
        session, hour_cst, features_dict
    """
    # Build features (we only care about X for live scoring; y is simulated labels)
    X, _y = build_strategy1(df_bars)

    if X.empty:
        return None

    # Take only the last row (most recently completed bar)
    last_row = X.iloc[[-1]]

    # Reindex to match the saved feature column order, fill missing with 0
    last_row = last_row.reindex(columns=feature_cols, fill_value=0)

    # Score
    prob = float(model.predict_proba(last_row)[0, 1])

    # Raw row values for signal details
    row_vals = last_row.iloc[0]
    direction = int(row_vals.get("direction", 1))

    # Entry price = close of the last bar in the raw OHLCV data
    entry_price = float(df_bars["close"].iloc[-1])

    # ATR for fallback SL/TP
    atr = _compute_atr(df_bars)

    # Try to use sl_distance and rr_ratio from features; fall back to ATR-based
    sl_distance = float(row_vals.get("sl_distance", 0.0))
    rr_ratio_feat = float(row_vals.get("rr_ratio", 0.0))

    if sl_distance > 0:
        sl_dist = sl_distance
    else:
        sl_dist = SL_ATR_MULT * atr

    if rr_ratio_feat > 0:
        rr = rr_ratio_feat
    else:
        rr = TP_ATR_MULT

    if direction == 1:   # long
        sl_price = entry_price - sl_dist
        tp_price = entry_price + rr * sl_dist
    else:                # short
        sl_price = entry_price + sl_dist
        tp_price = entry_price - rr * sl_dist

    rr_ratio = rr

    session_val = row_vals.get("session", 3)
    session_map_inv = {0: "asia", 1: "london", 2: "us", 3: "off"}
    session_label = session_map_inv.get(int(session_val), "off")

    hour_cst = int(row_vals.get("hour_cst", 0))

    features_dict = {col: float(row_vals[col]) for col in feature_cols}

    return {
        "prob": prob,
        "direction": direction,
        "entry_price": entry_price,
        "sl_price": sl_price,
        "tp_price": tp_price,
        "rr_ratio": rr_ratio,
        "session": session_label,
        "hour_cst": hour_cst,
        "features_dict": features_dict,
    }


# ---------------------------------------------------------------------------
# TP/SL check for pending trades
# ---------------------------------------------------------------------------

def check_pending_trades(df_bars: pd.DataFrame, symbol: str) -> None:
    """
    For each pending trade in the log matching `symbol`, check if the current
    bar's high/low crossed TP or SL, and close it if so.
    TP takes priority if both triggered in the same bar.
    """
    pending = get_pending_trades()

    if not pending.empty:
        pending = pending[pending["symbol"] == symbol]

    if pending.empty:
        print("[check] No pending trades to check.")
        return

    # Use the LAST bar for the check
    bar = df_bars.iloc[-1]
    bar_high = float(bar["high"])
    bar_low  = float(bar["low"])
    bar_time = bar.name if hasattr(bar, "name") else df_bars.index[-1]

    print(f"[check] Checking {len(pending)} pending trade(s) against bar "
          f"H={bar_high:.2f} L={bar_low:.2f}")

    for _, trade in pending.iterrows():
        trade_id   = trade["trade_id"]
        direction  = int(trade["direction"])
        entry_price = float(trade["entry_price"])
        sl_price   = float(trade["sl_price"])
        tp_price   = float(trade["tp_price"])
        symbol     = trade["symbol"]
        strategy   = trade["strategy"]

        tp_hit = False
        sl_hit = False

        if direction == 1:   # long
            tp_hit = bar_high >= tp_price
            sl_hit = bar_low  <= sl_price
        else:                 # short
            tp_hit = bar_low  <= tp_price
            sl_hit = bar_high >= sl_price

        if tp_hit:
            exit_price = tp_price
            label = 1
            outcome = "TP HIT"
        elif sl_hit:
            exit_price = sl_price
            label = 0
            outcome = "SL HIT"
        else:
            continue

        # Close the trade
        close_trade(
            trade_id=trade_id,
            exit_time=bar_time,
            exit_price=exit_price,
            label=label,
        )

        dir_str = "LONG" if direction == 1 else "SHORT"
        pnl = (exit_price - entry_price) * direction
        print()
        print("=" * 46)
        print(f"  TRADE CLOSED: {outcome}")
        print(f"  Trade ID : {trade_id}")
        print(f"  {symbol} {dir_str} ({strategy})")
        print(f"  Entry    : {entry_price:.2f}  Exit: {exit_price:.2f}")
        print(f"  PnL      : {pnl:+.2f} pts")
        print("=" * 46)


# ---------------------------------------------------------------------------
# Signal emission
# ---------------------------------------------------------------------------

def emit_signal(symbol: str, result: dict) -> str:
    """
    Log and print the trade signal.
    Returns the trade_id.
    """
    direction = result["direction"]
    entry_price = result["entry_price"]
    sl_price = result["sl_price"]
    tp_price = result["tp_price"]
    rr_ratio = result["rr_ratio"]
    prob = result["prob"]
    session = result["session"]
    hour_cst = result["hour_cst"]

    dir_str = "LONG" if direction == 1 else "SHORT"

    trade_id = log_trade(
        strategy=STRATEGY_NAME,
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        sl_price=sl_price,
        tp_price=tp_price,
        features=result["features_dict"],
    )

    # Write signal file for the automated scheduler to pick up
    tv_symbol = "MNQ1!" if symbol == "MNQ1" else "MES1!"
    signal_data = {
        "trade_id": trade_id,
        "symbol": symbol,
        "tv_symbol": tv_symbol,
        "direction": direction,
        "dir_str": dir_str,
        "entry_price": entry_price,
        "sl_price": sl_price,
        "tp_price": tp_price,
        "rr_ratio": rr_ratio,
        "prob": prob,
        "session": session,
        "hour_cst": hour_cst,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(SIGNAL_FILE, "w") as f:
        json.dump(signal_data, f, indent=2)

    print()
    print("=" * 46)
    print(f"  SIGNAL: {dir_str} {symbol} @ {entry_price:.2f}")
    print(f"  SL: {sl_price:.2f}  TP: {tp_price:.2f}  RR: {rr_ratio:.1f}")
    print(f"  Model confidence: {prob:.2f}")
    print(f"  Strategy: {STRATEGY_NAME}")
    print(f"  Session: {session}  Hour (CST): {hour_cst}")
    print(f"  Trade ID: {trade_id}")
    print(f"  Action: Enter {dir_str} on next 1H open or market order")
    print("=" * 46)
    print()

    return trade_id


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(symbol: str, threshold: float, check_only: bool) -> None:
    """Core execution logic."""
    yf_ticker = SYMBOL_TO_YF.get(symbol)
    if yf_ticker is None:
        raise ValueError(f"Unknown symbol '{symbol}'. Valid: {list(SYMBOL_TO_YF.keys())}")

    # Timestamp header
    now_utc = datetime.now(timezone.utc)
    print()
    print(f"[signal_engine] Run at {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"[signal_engine] Symbol: {symbol} ({yf_ticker}) | "
          f"Threshold: {threshold} | Check-only: {check_only}")
    print()

    # Fetch bars
    print(f"[data] Fetching 1H bars for {yf_ticker} (period=5d) ...")
    df_bars = fetch_bars(yf_ticker)
    print(f"[data] Got {len(df_bars)} bars. "
          f"Last bar close: {float(df_bars['close'].iloc[-1]):.2f} "
          f"at {df_bars.index[-1]}")
    print()

    # --- Check pending trades first ---
    check_pending_trades(df_bars, symbol)
    print()

    if check_only:
        print("[signal_engine] --check-only mode. Skipping signal generation.")
        return

    # --- Score the last bar ---
    print("[model] Loading model and features ...")
    model = load_model()
    feature_cols = load_feature_list()
    print(f"[model] Loaded. Feature count: {len(feature_cols)}")
    print()

    print("[model] Building features and scoring last bar ...")
    result = score_bar(df_bars, model, feature_cols)

    if result is None:
        print(f"[{STRATEGY_NAME}] No qualifying bar pattern found in data. "
              f"No signal this bar.")
        return

    prob = result["prob"]
    direction = result["direction"]
    dir_str = "LONG" if direction == 1 else "SHORT"

    print(f"[model] Last-bar score: prob={prob:.3f} | direction={dir_str} | "
          f"threshold={threshold}")
    print()

    if prob >= threshold:
        # EV gate: reject if model confidence can't overcome the R:R
        tp_dist = abs(result["tp_price"] - result["entry_price"])
        sl_dist = abs(result["sl_price"] - result["entry_price"])
        required_win_rate = sl_dist / (sl_dist + tp_dist)
        ev_pts = prob * tp_dist - (1 - prob) * sl_dist

        print(f"[ev_gate] R:R={result['rr_ratio']:.3f} | "
              f"Required win rate: {required_win_rate:.1%} | "
              f"Model prob: {prob:.1%} | "
              f"Expected value: {ev_pts:+.2f} pts")

        if ev_pts <= 0:
            print(f"[{STRATEGY_NAME}] BLOCKED by EV gate — negative expectancy "
                  f"({ev_pts:+.2f} pts). No signal this bar.")
            print()
        else:
            emit_signal(symbol, result)
    else:
        print(f"[{STRATEGY_NAME}] No signal this bar. Best prob: {prob:.3f}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DayTrading AI signal engine -- scores the last completed 1H bar."
    )
    parser.add_argument(
        "--symbol",
        default=DEFAULT_SYMBOL,
        choices=list(SYMBOL_TO_YF.keys()),
        help="Trading symbol (default: MNQ1)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Minimum model probability to emit a signal (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check pending trades for TP/SL; do not score for new signal",
    )

    args = parser.parse_args()

    try:
        run(
            symbol=args.symbol,
            threshold=args.threshold,
            check_only=args.check_only,
        )
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
