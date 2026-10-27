"""Published intraday strategies, coded as fixed rules (nothing is fitted to our data) so they can be
checked honestly on MES with real costs.

noise_area  Zarattini, Aziz & Barbon (2024), "Beat the Market: An Effective Intraday Momentum Strategy
            for S&P500 ETF (SPY)". Trade breakouts of a band built from the average move away from the
            open at that time of day over the last 14 sessions; trail with VWAP / the band; flat at close.
orb         Opening-range breakout in the style of Zarattini, Barbon & Aziz (2023): trade in the direction
            of the first N-minute candle, stop at the other side of that candle, target 10R, else exit at
            the close.

Both trade the regular session (09:30-16:00 ET) only and return one row per trade.
"""
import numpy as np
import pandas as pd

from .data import session_date
from .instruments import get_instrument

RTH_OPEN, RTH_CLOSE = 9 * 60 + 30, 16 * 60


def rth_minutes(bars):
    """Regular-session 1-minute bars with a session id and minute-of-day column."""
    mod = bars.index.hour * 60 + bars.index.minute
    rth = bars[(mod >= RTH_OPEN) & (mod < RTH_CLOSE)].copy()
    rth["mod"] = rth.index.hour * 60 + rth.index.minute
    rth["sdate"] = session_date(rth.index)
    # only complete days (a full session has 390 one-minute bars; allow a few missing)
    counts = rth.groupby("sdate")["close"].transform("size")
    return rth[counts >= 380]


def _trade(entry_time, exit_time, side, entry, exit_px, risk_pts, inst, cfg, reason):
    pnl_pts = side * (exit_px - entry)
    net = pnl_pts * inst.point_value - 2 * cfg.commission_per_side
    risk_usd = max(risk_pts, inst.tick_size) * inst.point_value
    return dict(entry_time=entry_time, exit_time=exit_time, side="long" if side > 0 else "short",
                entry=entry, exit=exit_px, pnl_pts=pnl_pts, net_usd=net, risk_usd=risk_usd, r=net / risk_usd,
                target_hit=reason == "target", exit_reason=reason, session=entry_time.normalize().tz_localize(None))


def noise_area(bars, cfg, lookback=14, mult=1.0, check_every=30):
    """Zarattini-Aziz-Barbon intraday momentum on 1 contract."""
    inst = get_instrument(cfg.instrument)
    slip = cfg.slippage_ticks * inst.tick_size
    r = rth_minutes(bars)
    day_open = r.groupby("sdate")["open"].transform("first")
    # sigma(t, minute) = mean over the previous `lookback` days of |close(minute) / open(day) - 1|
    move = (r["close"] / day_open - 1).abs()
    table = move.groupby([r["sdate"], r["mod"]]).last().unstack()          # days x minutes
    sigma = table.rolling(lookback, min_periods=lookback).mean().shift(1)   # previous days only
    prev_close = r.groupby("sdate")["close"].last().shift(1)

    trades = []
    for sdate, day in r.groupby("sdate"):
        if sdate not in sigma.index or sigma.loc[sdate].isna().all() or pd.isna(prev_close.get(sdate)):
            continue
        o = day["open"].iloc[0]
        pc = prev_close[sdate]
        sig = sigma.loc[sdate].reindex(day["mod"]).to_numpy() * mult
        ub = max(o, pc) * (1 + sig)
        lb = min(o, pc) * (1 - sig)
        tp = (day["high"] + day["low"] + day["close"]) / 3
        vwap = ((tp * day["volume"]).cumsum() / day["volume"].cumsum().replace(0, np.nan)).to_numpy()
        c = day["close"].to_numpy()
        nxt_open = np.append(day["open"].to_numpy()[1:], c[-1])
        mods = day["mod"].to_numpy()
        times = day.index
        pos, entry, entry_t, stop_at_entry = 0, 0.0, None, 0.0
        for k in range(len(day) - 1):
            bar_end = mods[k] + 1
            checkpoint = bar_end >= RTH_OPEN + 30 and (bar_end - RTH_OPEN) % check_every == 0
            if pos and checkpoint:
                trail = max(ub[k], vwap[k]) if pos > 0 else min(lb[k], vwap[k])
                if (pos > 0 and c[k] < trail) or (pos < 0 and c[k] > trail):
                    trades.append(_trade(entry_t, times[k + 1], pos, entry, nxt_open[k] - pos * slip,
                                         stop_at_entry, inst, cfg, "trail"))
                    pos = 0
            if not pos and checkpoint and bar_end <= RTH_CLOSE - 30 and np.isfinite(ub[k]):
                side = 1 if c[k] > ub[k] else -1 if c[k] < lb[k] else 0
                if side:
                    pos, entry, entry_t = side, nxt_open[k] + side * slip, times[k + 1]
                    stop_at_entry = abs(entry - (max(ub[k], vwap[k]) if side > 0 else min(lb[k], vwap[k])))
        if pos:
            trades.append(_trade(entry_t, times[-1], pos, entry, c[-1] - pos * slip, stop_at_entry, inst, cfg,
                                 "close"))
    return pd.DataFrame(trades)


def orb(bars, cfg, minutes=5, target_r=10.0):
    """Opening-range breakout: direction of the first `minutes` candle, stop at its far side,
    target `target_r` x risk, otherwise exit at the close. Stops are checked conservatively
    (stop first when a bar touches both)."""
    inst = get_instrument(cfg.instrument)
    slip = cfg.slippage_ticks * inst.tick_size
    r = rth_minutes(bars)
    trades = []
    for sdate, day in r.groupby("sdate"):
        first = day.iloc[:minutes]
        rest = day.iloc[minutes:]
        o, c = first["open"].iloc[0], first["close"].iloc[-1]
        if c == o or rest.empty:
            continue
        side = 1 if c > o else -1
        entry = rest["open"].iloc[0] + side * slip
        stop = first["low"].min() if side > 0 else first["high"].max()
        risk = abs(entry - stop)
        if risk < inst.tick_size:
            continue
        target = entry + side * target_r * risk
        hi, lo = rest["high"].to_numpy(), rest["low"].to_numpy()
        opn = rest["open"].to_numpy()
        exit_px, reason, exit_t = rest["close"].iloc[-1] - side * slip, "close", rest.index[-1]
        for k in range(len(rest)):
            stopped = lo[k] <= stop if side > 0 else hi[k] >= stop
            hit = hi[k] >= target + inst.tick_size if side > 0 else lo[k] <= target - inst.tick_size
            if stopped:
                fill = min(stop, opn[k]) if side > 0 else max(stop, opn[k])
                exit_px, reason, exit_t = fill - side * slip, "stop", rest.index[k]
                break
            if hit:
                exit_px, reason, exit_t = target, "target", rest.index[k]
                break
        trades.append(_trade(rest.index[0], exit_t, side, entry, exit_px, risk, inst, cfg, reason))
    return pd.DataFrame(trades)
