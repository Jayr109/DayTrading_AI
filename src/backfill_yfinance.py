"""
backfill_yfinance.py

One-time script to pull historical OHLCV data from Yahoo Finance and append
to existing CSVs in data/raw/, deduplicating by datetime index.

Timeframes pulled:
  - 5M  : 60 days  (period="60d", fallback "30d")
  - 15M : 60 days  (period="60d", fallback "30d")
  - 1H  : ~2 years (period="2y")

Tickers:
  - NQ=F  -> MNQ1
  - ES=F  -> MES1

Run from project root:
    python src/backfill_yfinance.py
"""

import os
import sys
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Config
# NOTE: No datetime.now() / system-clock usage — we rely entirely on yfinance
#       period= strings so Yahoo's server determines the date window.
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

SYMBOL_MAP = {
    "NQ=F": "MNQ1",
    "ES=F": "MES1",
}

# (yfinance interval, period_primary, period_fallback, csv_suffix)
TIMEFRAME_CONFIGS = [
    ("5m",  "60d", "30d", "5M"),
    ("15m", "60d", "30d", "15M"),
    ("1h",  "2y",  "2y",  "1H"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def csv_path(symbol_name: str, tf_suffix: str) -> str:
    return os.path.join(RAW_DIR, f"{symbol_name}_{tf_suffix}.csv")


def load_existing(path: str) -> pd.DataFrame:
    """Load an existing CSV or return an empty DataFrame."""
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, index_col="datetime", parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def fetch_yfinance(ticker: str, interval: str, period: str) -> pd.DataFrame:
    """Download data from yfinance using period= (no system clock), return UTC-indexed DataFrame."""
    raw = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        return pd.DataFrame()

    # Flatten multi-level columns if present (yfinance sometimes returns them)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Standardise column names
    raw.columns = [c.lower() for c in raw.columns]
    raw.index.name = "datetime"

    # Ensure UTC
    if raw.index.tz is None:
        raw.index = raw.index.tz_localize("UTC")
    else:
        raw.index = raw.index.tz_convert("UTC")

    # Keep only OHLCV columns
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in raw.columns]
    return raw[keep]


def merge_and_save(existing: pd.DataFrame, new_data: pd.DataFrame, path: str):
    """
    Combine existing and new data, deduplicate by datetime index,
    sort ascending, and save to CSV.

    Returns (total_bars, new_bars, date_range_str).
    """
    if existing.empty and new_data.empty:
        return 0, 0, "N/A"

    if existing.empty:
        combined = new_data.copy()
    elif new_data.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, new_data])

    # Deduplicate: keep last (new data wins on overlap)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)

    new_bars = 0
    if not existing.empty and not new_data.empty:
        existing_idx = set(existing.index)
        new_bars = sum(1 for i in new_data.index if i not in existing_idx)
    elif existing.empty:
        new_bars = len(combined)

    date_range = (
        f"{combined.index[0].strftime('%Y-%m-%d')} to "
        f"{combined.index[-1].strftime('%Y-%m-%d')}"
    )

    combined.index.name = "datetime"
    combined.to_csv(path)

    return len(combined), new_bars, date_range


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ensure_dir(RAW_DIR)

    print("=" * 60)
    print("Backfill: Yahoo Finance -> data/raw/")
    print("=" * 60)

    for ticker, symbol_name in SYMBOL_MAP.items():
        for interval, period_primary, period_fallback, tf_suffix in TIMEFRAME_CONFIGS:
            path = csv_path(symbol_name, tf_suffix)
            label = f"{symbol_name}_{tf_suffix}"

            # Load existing data
            existing = load_existing(path)

            # Fetch new data — try primary period, fall back if empty
            new_data = pd.DataFrame()
            for period in ([period_primary] if period_primary == period_fallback
                           else [period_primary, period_fallback]):
                try:
                    new_data = fetch_yfinance(ticker, interval, period)
                    if not new_data.empty:
                        print(f"  [{label}] fetched with period={period}")
                        break
                    print(f"  [{label}] period={period} returned no data, trying fallback...")
                except Exception as exc:
                    print(f"  [{label}] ERROR fetching (period={period}): {exc}")

            if new_data.empty:
                print(f"  [{label}] WARNING: yfinance returned no data for any period.")
                continue

            # Merge and save
            try:
                total, new_bars, date_range = merge_and_save(existing, new_data, path)
            except Exception as exc:
                print(f"  [{label}] ERROR saving: {exc}")
                continue

            print(
                f"  [{label}] total={total} bars | new={new_bars} bars | "
                f"range={date_range}"
            )

    print("=" * 60)
    print("Backfill complete.")


if __name__ == "__main__":
    main()
