"""Triple-barrier labelling: for every bar, simulate the scalp the bot would take on the NEXT bar's
open - a stop at k*ATR and a profit target at R times that distance - and record what happened.

Conservative fill rules, so the backtest does not flatter the strategy:
  * entry is a market order at the next bar's open plus slippage
  * if a bar touches both stop and target, the stop is assumed to have filled first
  * a limit target only fills if price trades at least one tick THROUGH it
  * a stop that is gapped through fills at the worse open price plus slippage
  * trades never cross the session break; they time out at max hold or at flatten time
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Outcomes:
    valid: np.ndarray        # a trade could be placed from this bar
    win: np.ndarray          # profit target hit
    exit_idx: np.ndarray     # bar on which the trade closed
    pnl_pts: np.ndarray      # points gained (after slippage), positive = profit
    stop_ticks: np.ndarray
    target_ticks: np.ndarray


def stop_target_ticks(atr_pts, tick, stop_mult, reward, min_ticks, max_ticks):
    stop = np.clip(np.round(stop_mult * np.asarray(atr_pts) / tick), min_ticks, max_ticks)
    stop = np.nan_to_num(stop, nan=min_ticks).astype(int)
    target = np.ceil(reward * stop - 1e-9).astype(int)
    return stop, target


def label(bars, aux, side, *, tick, stop_mult, reward, max_hold, minutes, flatten_min,
          min_ticks, max_ticks, slip_ticks):
    """side = +1 long / -1 short. bars/aux aligned DataFrames from features.build_features."""
    o = bars["open"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    c = bars["close"].to_numpy(float)
    sess = aux["sdate"].to_numpy()
    mod = aux["mod"].to_numpy()
    n = len(o)
    idx = np.arange(n)
    slip = slip_ticks * tick

    stop_t, tgt_t = stop_target_ticks(aux["atr"].to_numpy(float), tick, stop_mult, reward, min_ticks, max_ticks)
    entry_j = np.minimum(idx + 1, n - 1)
    can_hold = (mod + minutes) <= flatten_min        # bar finishes before flatten time

    valid = (idx + 1 < n) & (sess[entry_j] == sess) & can_hold[entry_j] & np.isfinite(aux["atr"].to_numpy(float))
    entry = o[entry_j] + side * slip
    stop_px = entry - side * stop_t * tick
    tgt_px = entry + side * tgt_t * tick

    done = ~valid
    win = np.zeros(n, bool)
    exit_idx = np.full(n, -1)
    exit_px = np.full(n, np.nan)
    last_ok = entry_j.copy()

    for k in range(1, max_hold + 1):
        j = idx + k
        inb = j < n
        jj = np.where(inb, j, n - 1)
        ok = ~done & inb & (sess[jj] == sess) & can_hold[jj]
        if not ok.any():
            break
        last_ok = np.where(ok, jj, last_ok)
        if side > 0:
            stopped = ok & (l[jj] <= stop_px)
            hit = ok & ~stopped & (h[jj] >= tgt_px + tick)
            stop_fill = np.minimum(stop_px, o[jj]) - slip
            tgt_fill = np.maximum(tgt_px, o[jj])
        else:
            stopped = ok & (h[jj] >= stop_px)
            hit = ok & ~stopped & (l[jj] <= tgt_px - tick)
            stop_fill = np.maximum(stop_px, o[jj]) + slip
            tgt_fill = np.minimum(tgt_px, o[jj])
        exit_px = np.where(stopped, stop_fill, np.where(hit, tgt_fill, exit_px))
        exit_idx = np.where(stopped | hit, jj, exit_idx)
        win |= hit
        done |= stopped | hit
        # trades still open when the session/flatten time runs out are closed in the timeout pass

    timed_out = valid & (exit_idx < 0)
    exit_idx = np.where(timed_out, last_ok, exit_idx)
    exit_px = np.where(timed_out, c[last_ok] - side * slip, exit_px)
    pnl = np.where(valid, side * (exit_px - entry), np.nan)
    return Outcomes(valid, win & valid, exit_idx, pnl, stop_t, tgt_t)
