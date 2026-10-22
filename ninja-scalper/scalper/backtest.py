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
        if risk > cfg.max_risk_per_trade_usd:
            continue
        net = out.pnl_pts[i] * inst.point_value - 2 * cfg.commission_per_side
        r = net / risk
        trades.append((index[min(i + 1, n - 1)], index[out.exit_idx[i]], "long" if long_side else "short",
                       float(best[i]), int(out.stop_ticks[i]), int(out.target_ticks[i]), bool(out.win[i]),
                       float(out.pnl_pts[i]), net, r, sdate[i]))
        day_count += 1
        day_r += r
        next_free = out.exit_idx[i]
    return pd.DataFrame(trades, columns=["entry_time", "exit_time", "side", "prob", "stop_ticks", "target_ticks",
                                         "target_hit", "pnl_pts", "net_usd", "r", "session"])


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


def account(trades, cfg, n_days, sims=2000, seed=7):
    """What these trades would have done to a small account trading 1 contract.

    Besides the actual sequence, the trades are reshuffled (bootstrapped) into many possible years
    to estimate how often an account of `starting_capital` drops below `min_account_usd` - the bad
    luck of a different order of wins and losses is a real risk with a ~30% win rate."""
    start, floor = cfg.starting_capital, cfg.min_account_usd
    if trades is None or len(trades) == 0:
        return {"start_usd": start, "end_usd": start, "low_usd": start, "wiped_out": False,
                "wipeout_prob_1y": 0.0, "median_1y_usd": start, "p10_1y_usd": start, "p90_1y_usd": start}
    usd = trades["net_usd"].to_numpy()
    bal = _stop_at_floor(start + np.cumsum(usd)[None, :], floor)[0]
    per_year = max(1, int(round(len(usd) / max(1, n_days) * 252)))
    rng = np.random.default_rng(seed)
    paths = _stop_at_floor(start + np.cumsum(rng.choice(usd, size=(sims, per_year), replace=True), axis=1), floor)
    busted = paths[:, -1] < floor
    ends = paths[:, -1]
    return {"start_usd": start, "end_usd": float(bal[-1]), "low_usd": float(min(start, bal.min())),
            "wiped_out": bool(bal[-1] < floor), "wipeout_prob_1y": float(busted.mean()),
            "median_1y_usd": float(np.median(ends)), "p10_1y_usd": float(np.percentile(ends, 10)),
            "p90_1y_usd": float(np.percentile(ends, 90))}


def _stop_at_floor(paths, floor):
    """A wiped-out account stops trading: hold each path at the balance where it first fell below the floor."""
    below = paths < floor
    hit = below.any(axis=1)
    first = below.argmax(axis=1)
    out = paths.copy()
    for k in np.flatnonzero(hit):
        out[k, first[k]:] = paths[k, first[k]]
    return out
