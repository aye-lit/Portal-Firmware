"""Command line:  python -m scalper <command> [options]

  download  get history (databento | ninjatrader | csv | yahoo | synthetic) into data/
  explore   test every timeframe / stop / reward:risk combination walk-forward -> reports/
  train     fit the live model (defaults to the leaderboard winner) -> models/
  learn     explore + train in one go
  retrain   pull new data (optional) and refit with the current live settings
  serve     run the bridge NinjaTrader connects to
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd

from . import data
from .config import Config
from .learn import REPORTS_DIR, MODELS_DIR, explore, train


def _cfg(a, **extra):
    over = {"instrument": a.instrument, **extra}
    if getattr(a, "timeframes", None):
        over["timeframes"] = a.timeframes
    if getattr(a, "quick", False):
        over.update(stop_atr_mults=[1.0], reward_ratios=[3.0], n_folds=3)
    return Config.load(a.config, **over)


def _expand(patterns):
    """Windows' cmd.exe does not expand wildcards, so do it here: "C:\\exports\\MES*.txt" works."""
    files = []
    for p in patterns or []:
        hits = sorted(glob.glob(p))
        files += hits or [p]
    return files


def cmd_download(a):
    sym = a.instrument
    a.files = _expand(a.files)
    if a.source == "databento":
        end = a.end or pd.Timestamp.now(tz="UTC").normalize().strftime("%Y-%m-%d")
        start = a.start or (pd.Timestamp(end) - pd.DateOffset(years=a.years)).strftime("%Y-%m-%d")
        df, minutes = data.download_databento(sym, start, end)
    elif a.source == "ninjatrader":
        df, minutes = data.load_ninjatrader_export(a.files, a.source_tz, bar_end=not a.bar_start), None
    elif a.source == "csv":
        df, minutes = data.load_csv(a.files, a.source_tz, bar_end=a.bar_end), None
    elif a.source == "yahoo":
        df, minutes = data.download_yahoo(sym, a.interval)
    else:
        df, minutes = data.synthetic_bars(days=a.days), 1
    minutes = minutes or data._infer_minutes(df)
    data.check_timezone(df)
    path, n = data.save_bars(df, sym, minutes, append=not a.replace)
    print(f"{len(df):,} new bars ({df.index[0]} -> {df.index[-1]}); {n:,} total stored in {path}")


def cmd_explore(a):
    cfg = _cfg(a)
    base, base_min = data.load_bars(cfg.instrument)
    print(f"Learning from {len(base):,} {base_min}-minute bars, {base.index[0].date()} -> {base.index[-1].date()}")
    board = explore(base, base_min, cfg)
    print(f"\nLeaderboard -> {os.path.join(REPORTS_DIR, 'leaderboard.csv')}")
    print(open(os.path.join(REPORTS_DIR, "summary.md")).read())
    return board, base, base_min, cfg


def _best_row():
    board = pd.read_csv(os.path.join(REPORTS_DIR, "leaderboard.csv"))
    return board.sort_values("score", ascending=False).iloc[0]


def cmd_train(a, base=None, base_min=None, cfg=None):
    cfg = cfg or _cfg(a)
    if base is None:
        base, base_min = data.load_bars(cfg.instrument)
    oos = None
    if a.timeframe is None:
        row = _best_row()
        tf, mult, rr = int(row.timeframe_min), float(row.stop_atr_mult), float(row.reward_risk)
        oos = {k: row[k] for k in ["trades", "win_rate", "avg_r", "profit_factor", "net_usd", "max_dd_usd", "t_stat",
                                   "folds_positive"]}
        if row.score < 2:
            print(f"CAUTION: best out-of-sample t-stat is {row.score:.2f} (< 2). The edge may be noise - "
                  "paper trade only.")
    else:
        tf, mult, rr = a.timeframe, a.stop_mult, a.reward
    if rr < 2:
        sys.exit("reward:risk must be at least 2:1")
    train(base, base_min, cfg, tf, mult, rr, oos=oos)


def cmd_learn(a):
    _, base, base_min, cfg = cmd_explore(a)
    a.timeframe = None
    cmd_train(a, base, base_min, cfg)


def cmd_retrain(a):
    meta_path = os.path.join(MODELS_DIR, f"{a.instrument}_live.json")
    meta = json.load(open(meta_path))
    if a.databento:
        stored, _ = data.load_bars(a.instrument)
        start = (stored.index[-1] - pd.Timedelta(days=1)).tz_convert("UTC").strftime("%Y-%m-%d")
        end = pd.Timestamp.now(tz="UTC").normalize().strftime("%Y-%m-%d")
        if start < end:
            df, minutes = data.download_databento(a.instrument, start, end)
            data.save_bars(df, a.instrument, minutes)
    cfg = Config.load(a.config, **{**meta["config"], "instrument": a.instrument}) if not a.config else _cfg(a)
    base, base_min = data.load_bars(cfg.instrument)
    train(base, base_min, cfg, meta["timeframe_min"], meta["stop_atr_mult"], meta["reward_risk"],
          oos=meta.get("oos_metrics"))


def cmd_serve(a):
    from .server import serve
    serve(a.model, a.host, a.port, a.instrument)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m scalper", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instrument", default="MES", help="MES, MNQ, M2K or MYM")
    ap.add_argument("--config", help="JSON file overriding scalper/config.py defaults")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download")
    d.add_argument("--source", choices=["databento", "ninjatrader", "csv", "yahoo", "synthetic"], required=True)
    d.add_argument("--files", nargs="+", help="export/CSV files (ninjatrader, csv)")
    d.add_argument("--years", type=int, default=5)
    d.add_argument("--start"); d.add_argument("--end")
    d.add_argument("--source-tz", default="America/New_York", help="time zone of timestamps in the files")
    d.add_argument("--bar-start", action="store_true", help="NinjaTrader files stamped with bar START time")
    d.add_argument("--bar-end", action="store_true", help="CSV files stamped with bar END time")
    d.add_argument("--interval", default="5m", help="yahoo: 1m or 5m")
    d.add_argument("--days", type=int, default=500, help="synthetic: sessions to generate")
    d.add_argument("--replace", action="store_true", help="overwrite stored data instead of merging")
    d.set_defaults(fn=cmd_download)

    for name, fn in [("explore", cmd_explore), ("learn", cmd_learn)]:
        e = sub.add_parser(name)
        e.add_argument("--timeframes", type=int, nargs="+", help="minutes, e.g. 1 2 3 5 10 15")
        e.add_argument("--quick", action="store_true", help="one stop size, 3:1 only, 3 folds")
        e.set_defaults(fn=fn, timeframe=None)

    t = sub.add_parser("train")
    t.add_argument("--timeframe", type=int, help="minutes; omit to use the leaderboard winner")
    t.add_argument("--stop-mult", type=float, default=1.0)
    t.add_argument("--reward", type=float, default=3.0)
    t.set_defaults(fn=cmd_train)

    r = sub.add_parser("retrain")
    r.add_argument("--databento", action="store_true", help="download bars since the last stored bar first")
    r.set_defaults(fn=cmd_retrain)

    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=5555)
    s.add_argument("--model")
    s.set_defaults(fn=cmd_serve)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
