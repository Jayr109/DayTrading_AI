"""
trade_logger.py

Logs model-generated trade signals and outcomes for the feedback loop.
Trade log lives at data/trade_log.csv relative to the project root.

Usage:
    from trade_logger import log_trade, close_trade, get_labeled_trades, \
                             get_pending_trades, trade_summary
"""

import os
import json
import uuid
import pandas as pd
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
_LOG_PATH = os.path.join(_PROJECT_ROOT, "data", "trade_log.csv")

# ---------------------------------------------------------------------------
# Column schema
# ---------------------------------------------------------------------------

COLUMNS = [
    "trade_id",
    "entry_time",
    "strategy",
    "symbol",
    "direction",
    "entry_price",
    "sl_price",
    "tp_price",
    "features_json",
    "exit_time",
    "exit_price",
    "label",
    "pnl_points",
]

VALID_STRATEGIES = {
    "strategy1_wick_rejection",
    "strategy2_orb_retest",
    "strategy3_zone_session",
}

VALID_SYMBOLS = {"MNQ1", "MES1"}
VALID_DIRECTIONS = {1, -1}
VALID_LABELS = {1, 0, -1}


# ---------------------------------------------------------------------------
# Internal I/O helpers
# ---------------------------------------------------------------------------

def _ensure_dir() -> None:
    os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)


def _load_log() -> pd.DataFrame:
    """Load the trade log CSV, or return an empty DataFrame with the correct schema."""
    _ensure_dir()
    if not os.path.exists(_LOG_PATH):
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(_LOG_PATH, dtype=str)
    # Ensure all expected columns exist
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[COLUMNS]


def _save_log(df: pd.DataFrame) -> None:
    _ensure_dir()
    df[COLUMNS].to_csv(_LOG_PATH, index=False)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_trade(
    strategy: str,
    symbol: str,
    direction: int,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    features: dict,
) -> str:
    """
    Append a new pending trade row to the log.

    Parameters
    ----------
    strategy  : one of VALID_STRATEGIES
    symbol    : "MNQ1" or "MES1"
    direction : 1 (long) or -1 (short)
    entry_price, sl_price, tp_price : float
    features  : dict of feature values at signal time

    Returns
    -------
    trade_id : str (UUID)
    """
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"strategy must be one of {VALID_STRATEGIES}, got {strategy!r}")
    if symbol not in VALID_SYMBOLS:
        raise ValueError(f"symbol must be one of {VALID_SYMBOLS}, got {symbol!r}")
    if direction not in VALID_DIRECTIONS:
        raise ValueError(f"direction must be 1 or -1, got {direction!r}")

    trade_id = str(uuid.uuid4())
    new_row = {
        "trade_id": trade_id,
        "entry_time": _now_iso(),
        "strategy": strategy,
        "symbol": symbol,
        "direction": str(int(direction)),
        "entry_price": str(float(entry_price)),
        "sl_price": str(float(sl_price)),
        "tp_price": str(float(tp_price)),
        "features_json": json.dumps(features),
        "exit_time": "",
        "exit_price": "",
        "label": "-1",
        "pnl_points": "",
    }

    df = _load_log()
    new_df = pd.DataFrame([new_row], columns=COLUMNS)
    df = pd.concat([df, new_df], ignore_index=True)
    _save_log(df)
    return trade_id


def close_trade(
    trade_id: str,
    exit_time,
    exit_price: float,
    label: int,
) -> None:
    """
    Fill exit fields for an open trade, and compute pnl_points.

    Parameters
    ----------
    trade_id   : UUID string returned by log_trade()
    exit_time  : ISO timestamp string or datetime object
    exit_price : float
    label      : 1 (TP hit), 0 (SL hit)
    """
    if label not in (0, 1):
        raise ValueError(f"label for a closed trade must be 0 or 1, got {label!r}")

    df = _load_log()
    mask = df["trade_id"] == trade_id
    if not mask.any():
        raise KeyError(f"trade_id not found: {trade_id!r}")

    idx = df.index[mask][0]

    # Resolve exit_time to ISO string
    if isinstance(exit_time, datetime):
        exit_time_str = exit_time.isoformat()
    else:
        exit_time_str = str(exit_time)

    entry_price = float(df.at[idx, "entry_price"])
    direction = int(df.at[idx, "direction"])
    pnl = (float(exit_price) - entry_price) * direction

    df.at[idx, "exit_time"] = exit_time_str
    df.at[idx, "exit_price"] = str(float(exit_price))
    df.at[idx, "label"] = str(int(label))
    df.at[idx, "pnl_points"] = str(round(pnl, 4))

    _save_log(df)


def get_labeled_trades() -> pd.DataFrame:
    """
    Return all closed trades (label != -1) with features unpacked into columns.

    The features_json column is parsed and each key becomes its own column.
    The original features_json column is retained as well.
    """
    df = _load_log()
    closed = df[df["label"].isin(["0", "1"])].copy()

    if closed.empty:
        return closed

    # Cast numeric columns
    for col in ["direction", "entry_price", "sl_price", "tp_price",
                "exit_price", "label", "pnl_points"]:
        closed[col] = pd.to_numeric(closed[col], errors="coerce")

    # Unpack features_json into individual columns
    def safe_parse(s):
        try:
            return json.loads(s) if pd.notna(s) and s != "" else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    features_expanded = closed["features_json"].apply(safe_parse).apply(pd.Series)
    # Drop any feature columns that duplicate existing top-level columns to
    # avoid duplicate column names which break boolean indexing in pandas.
    dup_cols = [c for c in features_expanded.columns if c in closed.columns]
    features_expanded = features_expanded.drop(columns=dup_cols)
    closed = pd.concat([closed.reset_index(drop=True), features_expanded.reset_index(drop=True)], axis=1)

    return closed


def get_pending_trades() -> pd.DataFrame:
    """Return all open trades (label == -1)."""
    df = _load_log()
    pending = df[df["label"] == "-1"].copy()
    for col in ["entry_price", "sl_price", "tp_price"]:
        pending[col] = pd.to_numeric(pending[col], errors="coerce")
    return pending


def trade_summary() -> None:
    """Print win rate, total trades, and avg PnL for closed trades."""
    closed = get_labeled_trades()
    pending = get_pending_trades()

    total_closed = len(closed)
    total_pending = len(pending)

    print("=" * 50)
    print("Trade Log Summary")
    print("=" * 50)
    print(f"  Pending trades : {total_pending}")
    print(f"  Closed trades  : {total_closed}")

    if total_closed == 0:
        print("  No closed trades yet.")
        print("=" * 50)
        return

    wins = int((closed["label"] == 1).sum())
    win_rate = wins / total_closed * 100
    avg_pnl = closed["pnl_points"].mean()

    print(f"  Win rate       : {win_rate:.1f}% ({wins}/{total_closed})")
    print(f"  Avg PnL (pts)  : {avg_pnl:.2f}")

    # Per-strategy breakdown
    print()
    print("  Per-strategy breakdown:")
    for strat in VALID_STRATEGIES:
        s_df = closed[closed["strategy"] == strat]
        if s_df.empty:
            continue
        s_wins = int((s_df["label"] == 1).sum())
        s_wr = s_wins / len(s_df) * 100
        s_avg = s_df["pnl_points"].mean()
        print(
            f"    {strat}: {len(s_df)} trades | "
            f"win rate {s_wr:.1f}% | avg pnl {s_avg:.2f} pts"
        )

    print("=" * 50)
