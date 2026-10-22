"""The learning loop.

explore(): for every timeframe x stop size x reward:risk combination, train gradient-boosted models
           that estimate P(target is hit before stop) for longs and shorts, and test them
           walk-forward - always trained on the past, scored on later data the model never saw.
           Produces a leaderboard that shows which timeframe scalps best.
train():   fit the final models for one combination on all history and save them for live trading.
"""
import json
import os
import time
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .backtest import account, entry_window, hhmm, metrics, simulate
from .data import resample
from .features import FEATURE_WARMUP_BARS, build_features
from .instruments import get_instrument
from .labels import label

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
REPORTS_DIR = os.path.join(ROOT, "reports")


@dataclass
class Prepared:
    minutes: int
    bars: pd.DataFrame
    X: pd.DataFrame
    aux: pd.DataFrame
    window: np.ndarray       # bars allowed to signal (entry window, warmed-up features)


def prepare(base, base_minutes, minutes, cfg):
    bars = resample(base, minutes) if minutes != base_minutes else base
    X, aux = build_features(bars, minutes)
    window = entry_window(aux, minutes, cfg)
    window[:FEATURE_WARMUP_BARS] = False
    return Prepared(minutes, bars, X, aux, window)


def max_hold_bars(cfg, minutes):
    return max(3, int(round(cfg.max_hold_minutes / minutes)))


def make_labels(p, cfg, inst, stop_mult, reward):
    kw = dict(tick=inst.tick_size, stop_mult=stop_mult, reward=reward, max_hold=max_hold_bars(cfg, p.minutes),
              minutes=p.minutes, flatten_min=hhmm(cfg.flatten_time), min_ticks=cfg.min_stop_ticks,
              max_ticks=cfg.max_stop_ticks, slip_ticks=cfg.slippage_ticks)
    return label(p.bars, p.aux, +1, **kw), label(p.bars, p.aux, -1, **kw)


def new_model(cfg):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=200, l2_regularization=1.0,
                                          random_state=cfg.random_state)


def fit(X, y, rows, cfg):
    """Fit on the masked rows (subsampled if huge). Returns None when there is nothing to learn."""
    idx = np.flatnonzero(rows)
    if len(idx) < 200 or y[idx].min() == y[idx].max():
        return None
    if len(idx) > cfg.max_train_rows:
        rng = np.random.default_rng(cfg.random_state)
        idx = np.sort(rng.choice(idx, cfg.max_train_rows, replace=False))
    m = new_model(cfg)
    m.fit(X.iloc[idx], y[idx])
    return m


def predict(model, X, rows):
    p = np.zeros(len(X))
    if model is not None and rows.any():
        p[rows] = model.predict_proba(X[rows])[:, 1]
    return p


def choose_threshold(aux, probs, outs, rows, cfg, inst):
    """Pick the probability cut-off that made the most R on calibration data (never on test data)."""
    best_t, best_r = None, -np.inf
    for t in cfg.prob_thresholds:
        tr = simulate(aux.index, aux, *probs, *outs, t, cfg, inst, rows=rows)
        if len(tr) < cfg.min_calib_trades:
            continue
        total = tr["r"].sum()
        if total > best_r:
            best_t, best_r = t, total
    return best_t


def date_splits(dates, cfg):
    """Expanding walk-forward: first `initial_train_frac` of sessions trains fold 1, the rest is cut
    into n_folds consecutive test windows."""
    n = len(dates)
    start = int(n * cfg.initial_train_frac)
    edges = np.linspace(start, n, cfg.n_folds + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            yield dates[:a], dates[a:b]


def walk_forward(p, cfg, inst, out_long, out_short):
    sd = p.aux["sdate"].to_numpy()
    dates = np.unique(sd[p.window])
    y_l, y_s = out_long.win.astype(int), out_short.win.astype(int)
    all_trades, fold_rows, n_test_days = [], [], 0
    for train_dates, test_dates in date_splits(dates, cfg):
        n_inner = int(len(train_dates) * (1 - cfg.calib_frac))
        inner = p.window & np.isin(sd, train_dates[:n_inner])
        calib = p.window & np.isin(sd, train_dates[n_inner:])
        test = p.window & np.isin(sd, test_dates)
        m_l = fit(p.X, y_l, inner & out_long.valid, cfg)
        m_s = fit(p.X, y_s, inner & out_short.valid, cfg)
        use = calib | test
        probs = (predict(m_l, p.X, use), predict(m_s, p.X, use))
        thr = choose_threshold(p.aux, probs, (out_long, out_short), calib, cfg, inst)
        n_test_days += len(test_dates)
        if thr is None:
            fold_rows.append({"test_start": str(test_dates[0])[:10], "threshold": None, "trades": 0, "total_r": 0.0})
            continue
        tr = simulate(p.aux.index, p.aux, *probs, out_long, out_short, thr, cfg, inst, rows=test)
        all_trades.append(tr)
        fold_rows.append({"test_start": str(test_dates[0])[:10], "threshold": thr, "trades": len(tr),
                          "total_r": float(tr["r"].sum()) if len(tr) else 0.0})
    trades = pd.concat(all_trades) if all_trades else pd.DataFrame()
    return trades, fold_rows, n_test_days


def explore(base, base_minutes, cfg, log=print):
    inst = get_instrument(cfg.instrument)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    rows, best_trades, best_score = [], None, -np.inf
    for minutes in cfg.timeframes:
        if minutes % base_minutes:
            log(f"skip {minutes}m: not a multiple of the {base_minutes}m source data")
            continue
        t0 = time.time()
        p = prepare(base, base_minutes, minutes, cfg)
        log(f"\n=== {minutes}-minute bars: {len(p.bars):,} bars, {int(p.window.sum()):,} tradable ===")
        for stop_mult in cfg.stop_atr_mults:
            for reward in cfg.reward_ratios:
                out_l, out_s = make_labels(p, cfg, inst, stop_mult, reward)
                trades, folds, n_days = walk_forward(p, cfg, inst, out_l, out_s)
                m = metrics(trades, n_days)
                acct = account(trades, cfg, n_days)
                pos_folds = sum(1 for f in folds if f["total_r"] > 0)
                score = m["t_stat"] if m["trades"] >= 30 else -np.inf
                row = {"timeframe_min": minutes, "stop_atr_mult": stop_mult, "reward_risk": reward,
                       **m, **acct, "folds_positive": f"{pos_folds}/{len(folds)}", "score": score,
                       "thresholds": [f["threshold"] for f in folds]}
                rows.append(row)
                log(f"  stop {stop_mult:>4}xATR  {reward}:1  trades {m['trades']:>5}  win {m['win_rate']:.0%}  "
                    f"avgR {m['avg_r']:+.3f}  PF {m['profit_factor']:.2f}  net ${m['net_usd']:>10,.0f}  "
                    f"t {m['t_stat']:+.2f}  folds+ {pos_folds}/{len(folds)}  "
                    f"${cfg.starting_capital:,.0f}->${acct['end_usd']:,.0f} (low ${acct['low_usd']:,.0f})")
                if score > best_score:
                    best_score, best_trades = score, trades.assign(timeframe_min=minutes)
        log(f"  ({time.time() - t0:.0f}s)")

    board = pd.DataFrame(rows).sort_values("score", ascending=False)
    board.to_csv(os.path.join(REPORTS_DIR, "leaderboard.csv"), index=False)
    if best_trades is not None and len(best_trades):
        best_trades.to_csv(os.path.join(REPORTS_DIR, "best_oos_trades.csv"), index=False)
    write_summary(board, cfg)
    return board


def write_summary(board, cfg):
    tf = board.groupby("timeframe_min").apply(lambda g: g.sort_values("score", ascending=False).iloc[0],
                                              include_groups=False).sort_values("score", ascending=False)
    lines = [f"# {cfg.instrument} scalping exploration (out-of-sample, walk-forward)", "",
             "Best setting per timeframe. All numbers are from data the model had NOT trained on, "
             f"after ${cfg.commission_per_side * 2:.2f} round-trip commission and {cfg.slippage_ticks} tick slippage.", "",
             f"Account columns: a ${cfg.starting_capital:,.0f} account trading 1 contract, skipping any signal that "
             f"risks more than ${cfg.max_risk_per_trade_usd:,.0f}. \"Wiped out\" = balance below "
             f"${cfg.min_account_usd:,.0f} (can no longer hold 1 contract); the 1-year odds come from reshuffling "
             "the trades into 2,000 possible years.", "",
             "| Timeframe | Stop | Target | Trades | Win % | Avg R | Net $ | t-stat | Folds + "
             f"| ${cfg.starting_capital:,.0f} became | Lowest balance | Chance of wipe-out in 1 yr | Typical 1-yr result |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for minutes, r in tf.iterrows():
        lines.append(f"| {minutes} min | {r.stop_atr_mult}x ATR | {r.reward_risk}:1 | {r.trades} | {r.win_rate:.0%} | "
                     f"{r.avg_r:+.3f} | {r.net_usd:,.0f} | {r.t_stat:+.2f} | {r.folds_positive} | "
                     f"${r.end_usd:,.0f} | ${r.low_usd:,.0f} | {r.wipeout_prob_1y:.0%} | "
                     f"${r.median_1y_usd:,.0f} (${r.p10_1y_usd:,.0f} to ${r.p90_1y_usd:,.0f}) |")
    top = board.iloc[0]
    lines += ["", f"**Best overall:** {top.timeframe_min}-minute bars, stop {top.stop_atr_mult}x ATR, "
                  f"target {top.reward_risk}:1 (t-stat {top.t_stat:+.2f}).", "",
              "A t-stat below ~2 means the edge is not statistically distinguishable from luck. "
              "Do not trade a configuration live unless it is positive in most folds and holds up in Sim/playback."]
    with open(os.path.join(REPORTS_DIR, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def train(base, base_minutes, cfg, minutes, stop_mult, reward, oos=None, log=print):
    """Fit the live models on all history. The threshold is chosen on the most recent
    `calib_frac` of sessions using models trained on everything before them; then the models
    are refit on all of it."""
    inst = get_instrument(cfg.instrument)
    p = prepare(base, base_minutes, minutes, cfg)
    out_l, out_s = make_labels(p, cfg, inst, stop_mult, reward)
    sd = p.aux["sdate"].to_numpy()
    dates = np.unique(sd[p.window])
    n_inner = int(len(dates) * (1 - cfg.calib_frac))
    inner = p.window & np.isin(sd, dates[:n_inner])
    calib = p.window & np.isin(sd, dates[n_inner:])
    y_l, y_s = out_l.win.astype(int), out_s.win.astype(int)
    m_l, m_s = fit(p.X, y_l, inner & out_l.valid, cfg), fit(p.X, y_s, inner & out_s.valid, cfg)
    probs = (predict(m_l, p.X, calib), predict(m_s, p.X, calib))
    thr = choose_threshold(p.aux, probs, (out_l, out_s), calib, cfg, inst)
    if thr is None:
        thr = max(cfg.prob_thresholds)
        log(f"warning: too few calibration trades; using the strictest threshold {thr}")
    calib_metrics = metrics(simulate(p.aux.index, p.aux, *probs, out_l, out_s, thr, cfg, inst, rows=calib),
                            len(dates) - n_inner)

    m_l = fit(p.X, y_l, p.window & out_l.valid, cfg)
    m_s = fit(p.X, y_s, p.window & out_s.valid, cfg)
    bundle = {
        "instrument": cfg.instrument, "timeframe_min": minutes, "stop_atr_mult": stop_mult, "reward_risk": reward,
        "max_hold_bars": max_hold_bars(cfg, minutes), "threshold": thr, "features": list(p.X.columns),
        "model_long": m_l, "model_short": m_s, "config": cfg.to_dict(),
        "trained_through": str(p.bars.index[-1]), "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "calibration_metrics": calib_metrics, "oos_metrics": oos,
    }
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{cfg.instrument}_live.joblib")
    joblib.dump(bundle, path)
    meta = {k: v for k, v in bundle.items() if not k.startswith("model_")}
    with open(path.replace(".joblib", ".json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    log(f"saved {path}  ({minutes}m, stop {stop_mult}xATR, {reward}:1, threshold {thr})")
    log(f"most-recent-period check: {calib_metrics['trades']} trades, avg R {calib_metrics['avg_r']:+.3f}, "
        f"net ${calib_metrics['net_usd']:,.0f}")
    return path
