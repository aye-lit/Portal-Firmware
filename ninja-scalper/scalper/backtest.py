"""Turn model probabilities into a realistic trade list: one position at a time, entry window,
max trades per day, daily loss limit, commissions and slippage. Then score it."""
import numpy as np
import pandas as pd


def hhmm(s):
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def entry_window(aux, minutes, cfg):
    """Bars whose CLOSE falls inside the trading window may generate a signal."""
    close_min = aux["mod"].to_numpy() + minutes
    return (close_min >= hhmm(cfg.trade_start)) & (close_min <= hhmm(cfg.trade_end))


def simulate(index, aux, p_long, p_short, out_long, out_short, threshold, cfg, inst, rows=None):
    """rows: optional boolean mask restricting which bars may generate signals (e.g. a test fold)."""
    n = len(index)
    p_long = np.where(out_long.valid, np.nan_to_num(p_long, nan=0.0), 0.0)
    p_short = np.where(out_short.valid, np.nan_to_num(p_short, nan=0.0), 0.0)
    best = np.maximum(p_long, p_short)
    cand = best >= threshold
    if rows is not None:
        cand &= rows
    sdate = aux["sdate"].to_numpy()

    trades = []
    next_free = -1
    day, day_count, day_r = None, 0, 0.0
    for i in np.flatnonzero(cand):
        if i < next_free:
            continue
        if sdate[i] != day:
            day, day_count, day_r = sdate[i], 0, 0.0
        if day_count >= cfg.max_trades_per_day or day_r <= -cfg.daily_loss_limit_r:
            continue
        long_side = p_long[i] >= p_short[i]
        out = out_long if long_side else out_short
        risk = out.stop_ticks[i] * inst.tick_value
        net = out.pnl_pts[i] * inst.point_value - 2 * cfg.commission_per_side
        r = net / risk
        trades.append((index[min(i + 1, n - 1)], index[out.exit_idx[i]], "long" if long_side else "short",
                       float(best[i]), int(out.stop_ticks[i]), int(out.target_ticks[i]), bool(out.win[i]),
                       float(out.pnl_pts[i]), net, r, risk, sdate[i]))
        day_count += 1
        day_r += r
        next_free = out.exit_idx[i]
    return pd.DataFrame(trades, columns=["entry_time", "exit_time", "side", "prob", "stop_ticks", "target_ticks",
                                         "target_hit", "pnl_pts", "net_usd", "r", "risk_usd",
                                         "session"])


def metrics(trades, n_days=None):
    if trades is None or len(trades) == 0:
        return {"trades": 0, "win_rate": 0.0, "target_rate": 0.0, "avg_r": 0.0, "total_r": 0.0, "net_usd": 0.0, "profit_factor": 0.0,
                "max_dd_usd": 0.0, "sharpe": 0.0, "t_stat": 0.0, "trades_per_day": 0.0, "avg_win_r": 0.0,
                "avg_loss_r": 0.0}
    r = trades["r"].to_numpy()
    usd = trades["net_usd"].to_numpy()
    eq = np.cumsum(usd)
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0], eq]))[1:] - eq)) if len(eq) else 0.0
    gains, losses = usd[usd > 0].sum(), -usd[usd < 0].sum()
    daily = trades.groupby("session")["net_usd"].sum()
    days = n_days or max(1, trades["session"].nunique())
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return {
        "trades": int(len(r)),
        "win_rate": float((r > 0).mean()),
        "target_rate": float(trades["target_hit"].mean()),
        "avg_r": float(r.mean()),
        "total_r": float(r.sum()),
        "net_usd": float(usd.sum()),
        "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
        "max_dd_usd": dd,
        "sharpe": float(daily.mean() / daily.std() * np.sqrt(252)) if len(daily) > 1 and daily.std() > 0 else 0.0,
        "t_stat": float(r.mean() / sd * np.sqrt(len(r))) if sd > 0 else 0.0,
        "trades_per_day": float(len(r) / days),
        "avg_win_r": float(r[r > 0].mean()) if (r > 0).any() else 0.0,
        "avg_loss_r": float(r[r <= 0].mean()) if (r <= 0).any() else 0.0,
    }


def account(trades, cfg, n_days, sims=1000, seed=7):
    """Replay per-contract trades on a compounding account that risks `risk_per_trade_pct` of its
    current balance on every trade (contracts = floor(balance * pct / risk of 1 contract); trades
    that cannot be sized to 1 contract are skipped).

    Besides the actual sequence, the trades are reshuffled (bootstrapped) into many possible years
    to show how much of the outcome is the luck of the order of wins and losses."""
    start, floor, pct = cfg.starting_capital, cfg.min_account_usd, cfg.risk_per_trade_pct / 100
    empty = {"start_usd": start, "end_usd": start, "low_usd": start, "wiped_out": False, "taken": 0,
             "skipped": 0, "capital_needed_usd": float("nan"), "wipeout_prob_1y": 0.0,
             "median_1y_usd": start, "p10_1y_usd": start, "p90_1y_usd": start}
    if trades is None or len(trades) == 0:
        return empty
    pnl = trades["net_usd"].to_numpy(float)
    risk = trades["risk_usd"].to_numpy(float)
    end, low, taken = _grow(pnl, risk, start, pct, floor)
    per_year = max(1, int(round(len(pnl) / max(1, n_days) * 252)))
    rng = np.random.default_rng(seed)
    ends = np.empty(sims)
    for k in range(sims):
        idx = rng.integers(0, len(pnl), per_year)
        ends[k] = _grow(pnl[idx], risk[idx], start, pct, floor)[0]
    return {"start_usd": start, "end_usd": end, "low_usd": low, "wiped_out": end < floor, "taken": taken,
            "skipped": len(pnl) - taken, "capital_needed_usd": float(np.median(risk) / pct),
            "wipeout_prob_1y": float((ends < floor).mean()), "median_1y_usd": float(np.median(ends)),
            "p10_1y_usd": float(np.percentile(ends, 10)), "p90_1y_usd": float(np.percentile(ends, 90))}


def _grow(pnl, risk, start, pct, floor):
    """(ending balance, lowest balance, trades taken) for one sequence; stops once below `floor`."""
    bal = low = start
    taken = 0
    for p, rk in zip(pnl, risk):
        n = int(bal * pct // rk) if rk > 0 else 0
        if n <= 0:
            continue
        bal += n * p
        taken += 1
        low = min(low, bal)
        if bal < floor:
            break
    return float(bal), float(low), taken
