"""
view_trades.py

View the trade log in a human-readable format.

Usage:
    python src/view_trades.py              # show all trades (pending + closed)
    python src/view_trades.py --pending    # show only open/pending trades
    python src/view_trades.py --closed     # show only closed trades
    python src/view_trades.py --summary    # show win rate + PnL summary only
    python src/view_trades.py --tail 10    # show last N trades
"""

import os
import sys
import argparse

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import pandas as pd
from trade_logger import _load_log, get_labeled_trades, get_pending_trades, trade_summary


def fmt_dir(d):
    try:
        return "LONG " if int(d) == 1 else "SHORT"
    except (ValueError, TypeError):
        return str(d)


def fmt_outcome(label):
    if str(label) == "1":
        return "WIN  (TP)"
    elif str(label) == "0":
        return "LOSS (SL)"
    elif str(label) == "-1":
        return "PENDING"
    return str(label)


def print_trade_table(df: pd.DataFrame, title: str) -> None:
    if df.empty:
        print(f"\n  {title}: none\n")
        return

    print(f"\n{'=' * 76}")
    print(f"  {title}  ({len(df)} trade{'s' if len(df) != 1 else ''})")
    print(f"{'=' * 76}")
    print(f"  {'#':<4} {'Time (UTC)':<22} {'Sym':<6} {'Dir':<6} {'Entry':>8} "
          f"{'SL':>8} {'TP':>8} {'PnL':>7}  Outcome")
    print(f"  {'-'*4} {'-'*22} {'-'*6} {'-'*6} {'-'*8} "
          f"{'-'*8} {'-'*8} {'-'*7}  {'-'*9}")

    for i, (_, row) in enumerate(df.iterrows(), 1):
        entry_t = str(row.get("entry_time", ""))[:19]
        symbol  = str(row.get("symbol", ""))
        direction = fmt_dir(row.get("direction", ""))
        entry_p = _flt(row.get("entry_price"))
        sl_p    = _flt(row.get("sl_price"))
        tp_p    = _flt(row.get("tp_price"))
        pnl     = row.get("pnl_points", "")
        pnl_str = f"{float(pnl):+.1f}" if pnl not in ("", None, "nan") else "   --"
        outcome = fmt_outcome(row.get("label", "-1"))

        print(f"  {i:<4} {entry_t:<22} {symbol:<6} {direction:<6} {entry_p:>8} "
              f"{sl_p:>8} {tp_p:>8} {pnl_str:>7}  {outcome}")

    print()


def _flt(val) -> str:
    try:
        return f"{float(val):.2f}"
    except (ValueError, TypeError):
        return "--"


def main():
    parser = argparse.ArgumentParser(
        description="View the DayTrading AI trade log"
    )
    parser.add_argument("--pending",  action="store_true", help="Show only pending trades")
    parser.add_argument("--closed",   action="store_true", help="Show only closed trades")
    parser.add_argument("--summary",  action="store_true", help="Show summary stats only")
    parser.add_argument("--tail",     type=int, default=0,  help="Show last N trades only")
    args = parser.parse_args()

    if args.summary:
        trade_summary()
        return

    raw = _load_log()

    if raw.empty:
        print("\n  No trades logged yet.\n")
        print("  Run the signal engine to start generating signals:")
        print("    python src/signal_engine.py --symbol MNQ1")
        return

    if args.tail > 0:
        raw = raw.tail(args.tail)

    if args.pending:
        df = raw[raw["label"] == "-1"].copy()
        print_trade_table(df, "PENDING TRADES")
    elif args.closed:
        df = raw[raw["label"].isin(["0", "1"])].copy()
        print_trade_table(df, "CLOSED TRADES")
    else:
        pending = raw[raw["label"] == "-1"].copy()
        closed  = raw[raw["label"].isin(["0", "1"])].copy()
        print_trade_table(pending, "PENDING TRADES")
        print_trade_table(closed,  "CLOSED TRADES")

    # Always show summary at the bottom
    print()
    trade_summary()


if __name__ == "__main__":
    main()
