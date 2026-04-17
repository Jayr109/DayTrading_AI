"""
collect_data.py

Pulls historical OHLCV bars from TradingView via the MCP and saves them to
data/raw/ as CSV files. This script is meant to be run interactively through
Claude Code, which has access to the TradingView MCP tools.

Usage:
    Run this module from a Claude Code session. Claude will call the MCP tools
    directly and write the resulting CSVs. This file defines the configuration
    and helper logic — the actual MCP calls are made by Claude.

Symbols:    MNQ1!, MES1!
Timeframes: 1H, 15M, 5M
Output:     data/raw/{symbol}_{timeframe}.csv
"""

import os
import pandas as pd
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["MNQ1!", "MES1!"]

TIMEFRAMES = {
    "1H":  "60",   # TradingView resolution string
    "15M": "15",
    "5M":  "5",
}

# How many bars to request per pull (TradingView MCP limit is typically 5000)
BARS_PER_PULL = 5000


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_bars(symbol: str, timeframe_label: str, bars: list[dict]) -> Path:
    """
    Convert a list of OHLCV bar dicts returned by data_get_ohlcv into a
    DataFrame and save to CSV.

    Expected bar dict keys (from TradingView MCP):
        time, open, high, low, close, volume

    Args:
        symbol:          e.g. "MNQ1!"
        timeframe_label: e.g. "1H"
        bars:            list of dicts from data_get_ohlcv

    Returns:
        Path to the saved CSV file.
    """
    df = pd.DataFrame(bars)

    # Normalise column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    # Parse timestamp — TradingView returns Unix seconds
    if "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.drop(columns=["time"])
        df = df.set_index("datetime").sort_index()

    # Sanitise symbol name for filename (MNQ1! → MNQ1)
    safe_symbol = symbol.replace("!", "")
    filename = f"{safe_symbol}_{timeframe_label}.csv"
    out_path = RAW_DIR / filename

    df.to_csv(out_path)
    print(f"Saved {len(df)} bars → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Load helper
# ---------------------------------------------------------------------------

def load_bars(symbol: str, timeframe_label: str) -> pd.DataFrame:
    """
    Load a previously saved OHLCV CSV from data/raw/.

    Returns a DataFrame indexed by UTC datetime with columns:
        open, high, low, close, volume
    """
    safe_symbol = symbol.replace("!", "")
    path = RAW_DIR / f"{safe_symbol}_{timeframe_label}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"No data file found at {path}. "
            f"Run data collection first via Claude + TradingView MCP."
        )

    df = pd.read_csv(path, index_col="datetime", parse_dates=True)
    df.index = df.index.tz_localize("UTC") if df.index.tzinfo is None else df.index
    return df


# ---------------------------------------------------------------------------
# Collection instructions for Claude
# ---------------------------------------------------------------------------

COLLECTION_GUIDE = """
DATA COLLECTION INSTRUCTIONS (for Claude Code + TradingView MCP)
=================================================================

For each symbol and timeframe below, run these MCP steps:

1. chart_set_symbol(symbol=<SYMBOL>)
2. chart_set_timeframe(timeframe=<RESOLUTION>)
3. bars = data_get_ohlcv(summary=False, count=5000)
4. Call save_bars(symbol=<SYMBOL>, timeframe_label=<LABEL>, bars=bars["bars"])

Collect:
  MNQ1! @ 1H  (resolution="60")
  MNQ1! @ 15M (resolution="15")
  MNQ1! @ 5M  (resolution="5")
  MES1! @ 1H  (resolution="60")
  MES1! @ 15M (resolution="15")
  MES1! @ 5M  (resolution="5")

Output files land in: data/raw/
"""


if __name__ == "__main__":
    print(COLLECTION_GUIDE)
    print("Available data files:")
    for f in sorted(RAW_DIR.glob("*.csv")):
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        print(f"  {f.name}: {len(df)} bars | {df.index[0]} → {df.index[-1]}")
