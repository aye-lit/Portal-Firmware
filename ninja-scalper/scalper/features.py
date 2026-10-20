"""Features the model learns from. Every value at bar i uses only bars <= i (no look-ahead), and the
same function runs on historical data during training and on live NinjaTrader bars in the server,
so what the model saw in training is exactly what it sees live.

Price distances are divided by ATR so the model learns shapes that hold across price levels and
volatility regimes (MES at 3,000 in 2022 behaves like MES at 6,000 in 2026)."""
import numpy as np
import pandas as pd

from .data import session_date

ATR_LEN = 14


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def atr(df, n=ATR_LEN):
    prev_close = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def build_features(df, minutes):
    """df: bars indexed by start time (US/Eastern). Returns (features DataFrame, aux DataFrame).

    aux holds non-feature columns the labeler/backtester needs (atr, session id, minute of day)."""
    o, h, l, c, v = (df[k] for k in ["open", "high", "low", "close", "volume"])
    a = atr(df)
    a_safe = a.replace(0, np.nan)
    f = pd.DataFrame(index=df.index)

    # momentum over several look-backs
    for n in (1, 3, 5, 10, 20):
        f[f"ret_{n}"] = (c - c.shift(n)) / a_safe

    # trend: distance to and slope of moving averages
    for n in (9, 21, 50):
        e = ema(c, n)
        f[f"dist_ema{n}"] = (c - e) / a_safe
        f[f"slope_ema{n}"] = (e - e.shift(3)) / a_safe
    f["ema9_vs_21"] = (ema(c, 9) - ema(c, 21)) / a_safe

    # oscillators / mean reversion
    f["rsi14"] = rsi(c, 14)
    f["rsi3"] = rsi(c, 3)
    m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_z"] = (c - m20) / s20.replace(0, np.nan)
    hh20, ll20 = h.rolling(20).max(), l.rolling(20).min()
    f["range_pos20"] = (c - ll20) / (hh20 - ll20).replace(0, np.nan)
    f["dist_hh20"] = (hh20 - c) / a_safe
    f["dist_ll20"] = (c - ll20) / a_safe

    # candle shape
    rng = (h - l).replace(0, np.nan)
    f["bar_range"] = (h - l) / a_safe
    f["body"] = (c - o) / rng
    f["upper_wick"] = (h - np.maximum(o, c)) / rng
    f["lower_wick"] = (np.minimum(o, c) - l) / rng
    f["body_prev"] = f["body"].shift()

    # volatility regime and volume
    f["atr_ratio"] = a / a.rolling(100, min_periods=20).mean()
    f["atr_bps"] = a / c * 1e4
    lv = np.log1p(v)
    f["vol_z"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std().replace(0, np.nan)

    # session context: VWAP, opening range, prior day levels, time of day
    sdate = session_date(df.index)
    g = pd.Series(sdate, index=df.index)
    tp = (h + l + c) / 3
    cum_pv = (tp * v).groupby(g).cumsum()
    cum_v = v.groupby(g).cumsum().replace(0, np.nan)
    vwap = cum_pv / cum_v
    f["dist_vwap"] = (c - vwap) / a_safe

    mod = df.index.hour * 60 + df.index.minute            # minute of day (bar start)
    bar_end = mod + minutes
    rth = (mod >= 9 * 60 + 30) & (mod < 16 * 60)
    # opening range = first 15 minutes of the cash session; only known once it has finished
    in_or = (mod >= 9 * 60 + 30) & (bar_end <= 9 * 60 + 45)
    or_hi = h.where(in_or).groupby(g).cummax()
    or_lo = l.where(in_or).groupby(g).cummin()
    or_done = pd.Series(bar_end >= 9 * 60 + 45, index=df.index)
    or_hi = or_hi.groupby(g).ffill().where(or_done)
    or_lo = or_lo.groupby(g).ffill().where(or_done)
    f["dist_or_hi"] = (c - or_hi) / a_safe
    f["dist_or_lo"] = (c - or_lo) / a_safe

    daily = df.groupby(g).agg(hi=("high", "max"), lo=("low", "min"), cl=("close", "last"))
    prev = daily.shift(1)
    f["dist_pdh"] = (c - g.map(prev["hi"])) / a_safe
    f["dist_pdl"] = (c - g.map(prev["lo"])) / a_safe
    f["dist_pdc"] = (c - g.map(prev["cl"])) / a_safe
    sess_hi = h.groupby(g).cummax()
    sess_lo = l.groupby(g).cummin()
    f["dist_sess_hi"] = (sess_hi - c) / a_safe
    f["dist_sess_lo"] = (c - sess_lo) / a_safe

    f["minute_of_day"] = mod
    f["mins_since_open"] = np.where(rth, mod - (9 * 60 + 30), -1)
    f["day_of_week"] = df.index.dayofweek

    f = f.replace([np.inf, -np.inf], np.nan).astype("float32")
    aux = pd.DataFrame({"atr": a, "sdate": sdate, "mod": mod}, index=df.index)
    return f, aux


FEATURE_WARMUP_BARS = 120  # rows before this have unreliable rolling values
