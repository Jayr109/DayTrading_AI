"""
features.py

Feature engineering for all three trading strategies.
Each strategy gets its own builder function that takes OHLCV DataFrames and
returns a feature matrix (X) and binary label (y: 1 = trade hit 3:1 TP, 0 = SL hit).

Strategies:
    1. 1H Wick Rejection       — both MNQ1 and MES1, all electronic hours
    2. 15M ORB Retest          — MNQ1 only, 8:30 AM–3:00 PM CST
    3. 7AM Zone + Session Levels — MES1 only, 8:30 AM–3:00 PM CST

Usage:
    from src.features import build_strategy1, build_strategy2, build_strategy3
    from src.collect_data import load_bars

    df_1h = load_bars("MNQ1!", "1H")
    X, y  = build_strategy1(df_1h)
"""

import numpy as np
import pandas as pd
import pytz

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CST = pytz.timezone("America/Chicago")

# US session window in CST
US_OPEN_H, US_OPEN_M   = 8,  30
US_CLOSE_H, US_CLOSE_M = 15,  0

# Asia session: 5 PM – 2 AM CST  (6 PM – 3 AM ET)
ASIA_OPEN_H,  ASIA_OPEN_M  = 17, 0
ASIA_CLOSE_H, ASIA_CLOSE_M = 2,  0   # next day

# London session: 2 AM – 10:30 AM CST  (3 AM – 11:30 AM ET)
LONDON_OPEN_H,  LONDON_OPEN_M  = 2,  0
LONDON_CLOSE_H, LONDON_CLOSE_M = 10, 30


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _to_cst(df: pd.DataFrame) -> pd.DataFrame:
    """Convert UTC-indexed DataFrame to CST."""
    df = df.copy()
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(CST)
    return df


def _candle_direction(o, c) -> int:
    """1 = bullish, -1 = bearish, 0 = doji."""
    if c > o:
        return 1
    elif c < o:
        return -1
    return 0


def _wick_to_body_ratio(o, h, l, c) -> tuple[float, float]:
    """Returns (upper_wick / body, lower_wick / body). Body floor = 1e-6."""
    body = max(abs(c - o), 1e-6)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    return upper_wick / body, lower_wick / body


def _simulate_trade(
    df: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    sl_price: float,
    tp_price: float,
) -> int:
    """
    Walk forward from entry_idx+1 bar-by-bar to see if TP or SL is hit first.
    Returns 1 (TP hit = win) or 0 (SL hit = loss). -1 = inconclusive (ran out of data).
    Uses high/low of each bar to determine which is hit first within a bar.
    """
    is_long = tp_price > entry_price

    for i in range(entry_idx + 1, len(df)):
        bar_high = df["high"].iloc[i]
        bar_low  = df["low"].iloc[i]

        if is_long:
            if bar_low <= sl_price:
                return 0
            if bar_high >= tp_price:
                return 1
        else:
            if bar_high >= sl_price:
                return 0
            if bar_low <= tp_price:
                return 1

    return -1  # inconclusive — drop from training set


# ---------------------------------------------------------------------------
# Strategy 1: 1H Wick Rejection
# ---------------------------------------------------------------------------

def build_strategy1(df_1h: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Feature matrix and labels for the 1H Wick Rejection strategy.

    Signal logic:
      - Bearish prior candle → current candle wicks below prior low, closes inside prior body → Long
      - Bullish prior candle → current candle wicks above prior high, closes inside prior body → Short

    Features engineered:
      - Prior candle: direction, body size, upper/lower wick ratios, range
      - Current candle: wick penetration depth, close position within prior body
      - Context: session label, rolling volatility (ATR), time of day
    """
    df = _to_cst(df_1h).copy()

    records = []

    for i in range(1, len(df) - 1):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        p_open, p_high, p_low, p_close = prev["open"], prev["high"], prev["low"], prev["close"]
        c_open, c_high, c_low, c_close = curr["open"], curr["high"], curr["low"], curr["close"]

        p_body_top    = max(p_open, p_close)
        p_body_bottom = min(p_open, p_close)
        p_direction   = _candle_direction(p_open, p_close)

        # --- Long setup: bearish prior candle, current wicks below and closes inside body ---
        if p_direction == -1:
            wick_below = c_low < p_low
            close_inside_body = p_body_bottom <= c_close <= p_body_top

            if wick_below and close_inside_body:
                entry = min(c_open, c_close)   # bottom of current candle body
                sl    = c_low - 1e-6           # just below wick
                tp    = p_body_top             # top of prior body

                label = _simulate_trade(df, i, entry, sl, tp)
                if label == -1:
                    continue

                uw_ratio, lw_ratio = _wick_to_body_ratio(p_open, p_high, p_low, p_close)
                penetration = (p_low - c_low) / max(abs(p_body_top - p_body_bottom), 1e-6)
                close_pct   = (c_close - p_body_bottom) / max(p_body_top - p_body_bottom, 1e-6)
                rr_ratio    = (tp - entry) / max(entry - sl, 1e-6)
                session     = _session_label(curr.name)

                records.append({
                    "timestamp":       curr.name,
                    "direction":       1,                         # long
                    "strategy":        1,
                    "p_direction":     p_direction,
                    "p_body_size":     p_body_top - p_body_bottom,
                    "p_range":         p_high - p_low,
                    "p_upper_wick_r":  uw_ratio,
                    "p_lower_wick_r":  lw_ratio,
                    "wick_penetration": penetration,
                    "close_pct_body":  close_pct,
                    "rr_ratio":        rr_ratio,
                    "session":         session,
                    "hour_cst":        curr.name.hour,
                    "label":           label,
                })

        # --- Short setup: bullish prior candle, current wicks above and closes inside body ---
        if p_direction == 1:
            wick_above = c_high > p_high
            close_inside_body = p_body_bottom <= c_close <= p_body_top

            if wick_above and close_inside_body:
                entry = max(c_open, c_close)   # top of current candle body
                sl    = c_high + 1e-6          # just above wick
                tp    = p_body_bottom          # bottom of prior body

                label = _simulate_trade(df, i, entry, sl, tp)
                if label == -1:
                    continue

                uw_ratio, lw_ratio = _wick_to_body_ratio(p_open, p_high, p_low, p_close)
                penetration = (c_high - p_high) / max(abs(p_body_top - p_body_bottom), 1e-6)
                close_pct   = (c_close - p_body_bottom) / max(p_body_top - p_body_bottom, 1e-6)
                rr_ratio    = (entry - tp) / max(sl - entry, 1e-6)
                session     = _session_label(curr.name)

                records.append({
                    "timestamp":       curr.name,
                    "direction":       -1,                        # short
                    "strategy":        1,
                    "p_direction":     p_direction,
                    "p_body_size":     p_body_top - p_body_bottom,
                    "p_range":         p_high - p_low,
                    "p_upper_wick_r":  uw_ratio,
                    "p_lower_wick_r":  lw_ratio,
                    "wick_penetration": penetration,
                    "close_pct_body":  close_pct,
                    "rr_ratio":        rr_ratio,
                    "session":         session,
                    "hour_cst":        curr.name.hour,
                    "label":           label,
                })

    result = pd.DataFrame(records).set_index("timestamp")
    X = result.drop(columns=["label"])
    y = result["label"]

    # Encode session as integer
    session_map = {"asia": 0, "london": 1, "us": 2, "off": 3}
    X["session"] = X["session"].map(session_map)

    return X, y


# ---------------------------------------------------------------------------
# Strategy 2: 15M ORB Retest (MNQ1, US session)
# ---------------------------------------------------------------------------

def build_strategy2(
    df_15m: pd.DataFrame,
    df_5m:  pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Feature matrix and labels for the 15M ORB Retest strategy.

    Signal logic:
      - Mark the 8:30 AM CST 15M candle as the opening range (OR).
      - On 5M chart: a close outside the OR marks a breakout.
      - First 5M candle that retests back into the OR after the breakout = entry.

    Features:
      - OR high, low, size, body size
      - Breakout direction, magnitude (close beyond OR / OR size)
      - Retest candle: how deep into OR price retested
      - Time elapsed from breakout to retest
      - Rolling ATR at breakout bar
    """
    df15 = _to_cst(df_15m).copy()
    df5  = _to_cst(df_5m).copy()

    records = []

    # Group 15M bars by trading date
    trading_dates = df15.index.normalize().unique()

    for date in trading_dates:
        # Find the 8:30 AM CST 15M candle for this date
        day_bars_15 = df15[df15.index.normalize() == date]
        or_bars = day_bars_15[
            (day_bars_15.index.hour == US_OPEN_H) &
            (day_bars_15.index.minute == US_OPEN_M)
        ]
        if or_bars.empty:
            continue

        or_bar = or_bars.iloc[0]
        or_high  = or_bar["high"]
        or_low   = or_bar["low"]
        or_open  = or_bar["open"]
        or_close = or_bar["close"]
        or_size  = or_high - or_low
        or_body  = abs(or_close - or_open)

        if or_size < 1e-6:
            continue

        # 5M bars for this day after the OR close (8:45 AM CST onward)
        or_end_time = or_bars.index[0] + pd.Timedelta(minutes=15)
        session_end = or_bars.index[0].replace(
            hour=US_CLOSE_H, minute=US_CLOSE_M, second=0, microsecond=0
        )
        day_5m = df5[
            (df5.index >= or_end_time) &
            (df5.index <= session_end)
        ]

        if day_5m.empty:
            continue

        # Find first breakout close
        breakout_idx  = None
        breakout_dir  = None
        breakout_mag  = 0.0

        for j, (ts, bar) in enumerate(day_5m.iterrows()):
            if bar["close"] > or_high:
                breakout_idx = j
                breakout_dir = 1   # bullish
                breakout_mag = (bar["close"] - or_high) / or_size
                break
            elif bar["close"] < or_low:
                breakout_idx = j
                breakout_dir = -1  # bearish
                breakout_mag = (or_low - bar["close"]) / or_size
                break

        if breakout_idx is None:
            continue

        # Find first retest into OR after breakout
        post_breakout = day_5m.iloc[breakout_idx + 1:]

        retest_bar = None
        retest_depth = 0.0
        bars_to_retest = 0

        for k, (ts, bar) in enumerate(post_breakout.iterrows()):
            if breakout_dir == 1:
                # Bullish breakout → retest = price comes back into OR from above
                if bar["low"] <= or_high:
                    retest_bar   = bar
                    retest_depth = (or_high - bar["low"]) / or_size
                    bars_to_retest = k + 1
                    break
            else:
                # Bearish breakout → retest = price comes back into OR from below
                if bar["high"] >= or_low:
                    retest_bar   = bar
                    retest_depth = (bar["high"] - or_low) / or_size
                    bars_to_retest = k + 1
                    break

        if retest_bar is None:
            continue

        # Entry, SL, TP
        retest_idx_in_day5m = breakout_idx + 1 + bars_to_retest

        if breakout_dir == 1:
            entry = or_high
            # SL: prior 5M swing low before retest candle
            prior_lows = day_5m.iloc[:retest_idx_in_day5m]["low"]
            sl = prior_lows.min()
            tp = entry + 3.0 * (entry - sl)
        else:
            entry = or_low
            prior_highs = day_5m.iloc[:retest_idx_in_day5m]["high"]
            sl = prior_highs.max()
            tp = entry - 3.0 * (sl - entry)

        label = _simulate_trade(
            day_5m,
            retest_idx_in_day5m,
            entry, sl, tp
        )
        if label == -1:
            continue

        # ATR proxy: rolling std of last 14 bar ranges
        recent_ranges = day_5m["high"].iloc[:retest_idx_in_day5m] - \
                        day_5m["low"].iloc[:retest_idx_in_day5m]
        atr = recent_ranges.tail(14).mean() if len(recent_ranges) >= 3 else np.nan

        records.append({
            "timestamp":       retest_bar.name,
            "direction":       breakout_dir,
            "strategy":        2,
            "or_size":         or_size,
            "or_body_ratio":   or_body / max(or_size, 1e-6),
            "breakout_mag":    breakout_mag,
            "retest_depth":    retest_depth,
            "bars_to_retest":  bars_to_retest,
            "sl_distance":     abs(entry - sl),
            "atr":             atr,
            "hour_cst":        retest_bar.name.hour,
            "minute_cst":      retest_bar.name.minute,
            "label":           label,
        })

    result = pd.DataFrame(records).set_index("timestamp")
    X = result.drop(columns=["label"])
    y = result["label"]
    return X, y


# ---------------------------------------------------------------------------
# Strategy 3: 7AM Zone + Session Levels (MES1, US session)
# ---------------------------------------------------------------------------

def build_strategy3(
    df_15m: pd.DataFrame,
    df_5m:  pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Feature matrix and labels for the 7AM Zone + Session Levels strategy.

    Signal logic:
      - Mark the 7:00–7:15 AM CST 15M candle as the zone.
      - After 8:30 AM: wait for a 5M close ≥5 pts outside the zone.
      - Retest into zone (top, bottom, or midpoint) → entry on next candle.

    Features:
      - Zone size, body ratio
      - Breakout magnitude (pts beyond zone)
      - Entry type: top / bottom / midpoint
      - Session levels: distance to nearest Asia/London high or low
      - Bars from breakout to retest
      - ATR at time of entry
    """
    df15 = _to_cst(df_15m).copy()
    df5  = _to_cst(df_5m).copy()

    records = []
    BREAKOUT_MIN_PTS = 5.0

    trading_dates = df15.index.normalize().unique()

    for date in trading_dates:
        day_bars_15 = df15[df15.index.normalize() == date]

        # 7AM zone
        zone_bars = day_bars_15[
            (day_bars_15.index.hour == 7) &
            (day_bars_15.index.minute == 0)
        ]
        if zone_bars.empty:
            continue

        zone_bar  = zone_bars.iloc[0]
        zone_high = zone_bar["high"]
        zone_low  = zone_bar["low"]
        zone_mid  = (zone_high + zone_low) / 2
        zone_size = zone_high - zone_low
        zone_body = abs(zone_bar["close"] - zone_bar["open"])

        if zone_size < 1e-6:
            continue

        # Session levels for this date (prior day Asia + London)
        asia_high, asia_low, london_high, london_low = _get_session_levels(df5, date)

        # 5M bars: 8:30 AM – 3:00 PM CST
        session_start = day_bars_15.index[0].replace(
            hour=US_OPEN_H, minute=US_OPEN_M, second=0, microsecond=0
        )
        session_end = day_bars_15.index[0].replace(
            hour=US_CLOSE_H, minute=US_CLOSE_M, second=0, microsecond=0
        )
        day_5m = df5[
            (df5.index >= session_start) &
            (df5.index <= session_end)
        ]
        if day_5m.empty:
            continue

        # Find first confirmed breakout (close ≥5 pts outside zone)
        breakout_idx = None
        breakout_dir = None
        breakout_mag = 0.0

        for j, (ts, bar) in enumerate(day_5m.iterrows()):
            if bar["close"] > zone_high + BREAKOUT_MIN_PTS:
                breakout_idx = j
                breakout_dir = 1
                breakout_mag = bar["close"] - zone_high
                break
            elif bar["close"] < zone_low - BREAKOUT_MIN_PTS:
                breakout_idx = j
                breakout_dir = -1
                breakout_mag = zone_low - bar["close"]
                break

        if breakout_idx is None:
            continue

        # Find retest: price returns to zone top, bottom, or midpoint
        post_breakout  = day_5m.iloc[breakout_idx + 1:]
        retest_bar     = None
        retest_type    = None   # "top", "bottom", "midpoint"
        bars_to_retest = 0

        for k, (ts, bar) in enumerate(post_breakout.iterrows()):
            if breakout_dir == 1:
                if bar["low"] <= zone_high:
                    retest_bar     = bar
                    retest_type    = "top"
                    bars_to_retest = k + 1
                    break
                elif bar["low"] <= zone_mid:
                    retest_bar     = bar
                    retest_type    = "midpoint"
                    bars_to_retest = k + 1
                    break
            else:
                if bar["high"] >= zone_low:
                    retest_bar     = bar
                    retest_type    = "bottom"
                    bars_to_retest = k + 1
                    break
                elif bar["high"] >= zone_mid:
                    retest_bar     = bar
                    retest_type    = "midpoint"
                    bars_to_retest = k + 1
                    break

        if retest_bar is None:
            continue

        # Entry on the NEXT candle after confirmation
        retest_pos = breakout_idx + 1 + bars_to_retest
        if retest_pos + 1 >= len(day_5m):
            continue
        entry_bar = day_5m.iloc[retest_pos + 1]

        # Entry price and SL
        if breakout_dir == 1:
            entry = zone_high if retest_type == "top" else zone_mid
            sl    = entry - 8.0   # default 8 pts, scaled to zone below
            sl    = min(sl, entry - max(zone_size * 0.5, 5.0))
            sl    = max(sl, entry - 12.0)  # cap at 12 pts
        else:
            entry = zone_low if retest_type == "bottom" else zone_mid
            sl    = entry + 8.0
            sl    = max(sl, entry + max(zone_size * 0.5, 5.0))
            sl    = min(sl, entry + 12.0)

        tp = entry + 3.0 * (entry - sl) if breakout_dir == 1 else entry - 3.0 * (sl - entry)

        label = _simulate_trade(day_5m, retest_pos + 1, entry, sl, tp)
        if label == -1:
            continue

        # Distance to nearest session level
        all_levels = [x for x in [asia_high, asia_low, london_high, london_low] if x is not None]
        dist_to_level = min(abs(entry - lvl) for lvl in all_levels) if all_levels else np.nan

        recent_ranges = day_5m["high"].iloc[:retest_pos] - day_5m["low"].iloc[:retest_pos]
        atr = recent_ranges.tail(14).mean() if len(recent_ranges) >= 3 else np.nan

        records.append({
            "timestamp":        entry_bar.name,
            "direction":        breakout_dir,
            "strategy":         3,
            "zone_size":        zone_size,
            "zone_body_ratio":  zone_body / max(zone_size, 1e-6),
            "breakout_mag":     breakout_mag,
            "retest_type":      {"top": 0, "bottom": 1, "midpoint": 2}[retest_type],
            "bars_to_retest":   bars_to_retest,
            "sl_distance":      abs(entry - sl),
            "dist_to_session_level": dist_to_level,
            "atr":              atr,
            "hour_cst":         entry_bar.name.hour,
            "minute_cst":       entry_bar.name.minute,
            "label":            label,
        })

    result = pd.DataFrame(records).set_index("timestamp")
    X = result.drop(columns=["label"])
    y = result["label"]
    return X, y


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _session_label(ts: pd.Timestamp) -> str:
    """Return session label for a CST timestamp."""
    h = ts.hour
    m = ts.minute
    t = h * 60 + m

    us_open  = US_OPEN_H  * 60 + US_OPEN_M
    us_close = US_CLOSE_H * 60 + US_CLOSE_M
    lon_open  = LONDON_OPEN_H  * 60 + LONDON_OPEN_M
    lon_close = LONDON_CLOSE_H * 60 + LONDON_CLOSE_M

    if us_open <= t < us_close:
        return "us"
    elif lon_open <= t < lon_close:
        return "london"
    elif t >= ASIA_OPEN_H * 60 or t < ASIA_CLOSE_H * 60:
        return "asia"
    return "off"


def _get_session_levels(
    df_5m: pd.DataFrame, date: pd.Timestamp
) -> tuple[float | None, float | None, float | None, float | None]:
    """
    Return (asia_high, asia_low, london_high, london_low) for the given trading date
    by scanning the prior session bars.
    """
    prior = df_5m[df_5m.index.normalize() < date]
    if prior.empty:
        return None, None, None, None

    # Asia: previous day 5 PM – 2 AM CST
    prev_date = date - pd.Timedelta(days=1)
    asia = prior[
        (prior.index.normalize() == prev_date) &
        (
            (prior.index.hour >= ASIA_OPEN_H) |
            (prior.index.hour < ASIA_CLOSE_H)
        )
    ]

    # London: same date 2 AM – 10:30 AM CST (but prior to US open)
    london = prior[
        (prior.index.normalize() == date) &
        (prior.index.hour >= LONDON_OPEN_H) &
        (
            (prior.index.hour < LONDON_CLOSE_H) |
            ((prior.index.hour == LONDON_CLOSE_H) & (prior.index.minute <= LONDON_CLOSE_M))
        )
    ]

    asia_high   = float(asia["high"].max())   if not asia.empty   else None
    asia_low    = float(asia["low"].min())    if not asia.empty   else None
    london_high = float(london["high"].max()) if not london.empty else None
    london_low  = float(london["low"].min())  if not london.empty else None

    return asia_high, asia_low, london_high, london_low
