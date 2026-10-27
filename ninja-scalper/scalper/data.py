"""Getting price history into one clean, continuous bar series.

Internal convention: a DataFrame indexed by the bar's START time (tz-aware, US/Eastern) with
float columns open/high/low/close/volume. Everything else in the project relies on it.

Sources:
  * Databento   - 5+ years of CME 1-minute bars over the internet (paid API, cheap; free credit on signup)
  * NinjaTrader - Tools > Historical Data > Export (free with your NinjaTrader data feed)
  * Any CSV     - FirstRate Data, Kibot, Portara, etc.
  * Yahoo       - free but only ~60 days of 5-minute / ~7 days of 1-minute data; good for a smoke test
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

from .instruments import get_instrument

COLS = ["open", "high", "low", "close", "volume"]
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# ----------------------------------------------------------------------------------- storage

def bars_path(symbol, minutes):
    return os.path.join(DATA_DIR, f"{symbol.upper()}_{minutes}min.csv.gz")


def save_bars(df, symbol, minutes, append=True):
    """Save bars, merging with whatever is already stored so history only ever grows."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = bars_path(symbol, minutes)
    if append and os.path.exists(path):
        old = read_bars_file(path)
        df = pd.concat([old, df])
        df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    out = df[COLS].copy()
    out.index = out.index.tz_convert("UTC")
    out.index.name = "time_utc"
    out.to_csv(path, compression="gzip")
    return path, len(df)


def read_bars_file(path, tz="America/New_York"):
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(tz)
    return df[COLS].astype(float)


def load_bars(symbol, tz="America/New_York"):
    """Return (bars, base_minutes) using the finest resolution stored for this symbol."""
    found = []
    for p in glob.glob(os.path.join(DATA_DIR, f"{symbol.upper()}_*min.csv.gz")):
        m = re.search(r"_(\d+)min\.csv\.gz$", p)
        if m:
            found.append((int(m.group(1)), p))
    if not found:
        raise FileNotFoundError(
            f"No stored bars for {symbol} in {DATA_DIR}. Run `python -m scalper download ...` first.")
    minutes, path = min(found)
    return read_bars_file(path, tz), minutes


# ----------------------------------------------------------------------------------- helpers

def session_date(index):
    """CME trading day: the Globex session that opens at 18:00 ET belongs to the next calendar day."""
    return (index + pd.Timedelta(hours=6)).normalize().tz_localize(None)


def resample(df, minutes):
    if minutes == 1 and len(df) and _infer_minutes(df) == 1:
        return df
    rule = f"{minutes}min"
    out = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return out.dropna(subset=["open"])


def _infer_minutes(df):
    d = pd.Series(df.index).diff().dropna()
    if d.empty:
        return 1
    return max(1, int(round(d.mode().iloc[0].total_seconds() / 60)))


def clean(df):
    df = df[COLS].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    # stray weekend prints (a bar or two on Saturday/Sunday outside any Globex session)
    df = df[session_date(df.index).dayofweek < 5]
    # repair occasional bad prints where high/low do not contain open/close
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


def check_timezone(df):
    """The busiest minute of the day should be the 09:30 ET cash open (or the 16:00 close).
    If it is not, the source time zone was probably set wrong - warn loudly."""
    if len(df) < 2000:
        return
    vol = df["volume"].groupby(df.index.hour * 60 + df.index.minute).mean()
    peak = int(vol.idxmax())
    if not (9 * 60 + 25 <= peak <= 10 * 60 + 5 or 15 * 60 + 45 <= peak <= 16 * 60):
        print(f"WARNING: busiest bar of the day starts at {peak // 60:02d}:{peak % 60:02d} ET. "
              "Expected ~09:30 ET. Check --source-tz / --bar-end for this data.", file=sys.stderr)


def back_adjust(df, contract):
    """Stitch per-contract data into one continuous series.

    Each session uses the contract that traded the most volume that day (never rolling back to
    an older contract). Earlier contracts are shifted by the price gap at each roll (Panama
    adjustment) so the stitched series has no artificial jumps.
    """
    df = df.assign(contract=contract, sdate=session_date(df.index))
    names = sorted(set(contract), key=_contract_sort_key)
    rank = {c: i for i, c in enumerate(names)}
    vol = df.groupby(["sdate", "contract"])["volume"].sum().unstack(fill_value=0)
    chosen = vol.idxmax(axis=1).map(rank).cummax().map(dict(enumerate(names)))

    keep = df["contract"].values == chosen.reindex(df["sdate"]).values
    out = df[keep].copy()
    adj = np.zeros(len(out))
    prev = chosen.shift()
    for sdate in chosen.index[(chosen != prev) & prev.notna()]:
        before = df["sdate"] < sdate
        old = df.loc[before & (df["contract"] == prev[sdate]), "close"]
        new = df.loc[before & (df["contract"] == chosen[sdate]), "close"]
        common = old.index.intersection(new.index)
        if len(common):
            t = common.max()
            adj[(out["sdate"] < sdate).values] += float(new[t] - old[t])
    for c in ["open", "high", "low", "close"]:
        out[c] = out[c] + adj
    return out[COLS]


def _contract_sort_key(name):
    """'MES 03-24' / 'MESH4' / filename -> sortable (year, month); unknown names sort by text."""
    m = re.search(r"(\d{2})-(\d{2})", name)
    if m:
        return (int(m.group(2)), int(m.group(1)), name)
    codes = "FGHJKMNQUVXZ"
    m = re.search(r"([FGHJKMNQUVXZ])(\d{1,2})(?!.*[FGHJKMNQUVXZ]\d)", name)
    if m:
        return (int(m.group(2)) % 100, codes.index(m.group(1)) + 1, name)
    return (9999, 99, name)


# ----------------------------------------------------------------------------------- sources

def load_ninjatrader_export(paths, source_tz="America/New_York", bar_end=True):
    """NinjaTrader 8 export format: 'yyyyMMdd HHmmss;open;high;low;close;volume'.

    NinjaTrader stamps minute bars with their CLOSE time; we shift to the start time.
    Export one file per contract (e.g. 'MES 03-25.Last.txt') and pass them all; they are rolled
    and back-adjusted into one continuous series.
    """
    frames, contracts = [], []
    for p in paths:
        raw = pd.read_csv(p, sep=";", header=None, names=["ts"] + COLS, dtype={"ts": str})
        idx = pd.to_datetime(raw["ts"], format="%Y%m%d %H%M%S")
        f = raw[COLS].set_index(pd.DatetimeIndex(idx))
        f.index = f.index.tz_localize(source_tz, ambiguous="NaT", nonexistent="NaT")
        f = f[f.index.notna()]
        f = f[~f.index.duplicated(keep="last")]
        if bar_end:
            f.index = f.index - pd.Timedelta(minutes=_infer_minutes(f))
        f.index = f.index.tz_convert("America/New_York")
        frames.append(f)
        contracts.append(os.path.basename(p).split(".")[0])
    if len(frames) == 1:
        return clean(frames[0])
    tagged = pd.concat([f.assign(_c=c) for f, c in zip(frames, contracts)]).sort_index()
    return clean(back_adjust(tagged[COLS], tagged["_c"].values))


def load_csv(paths, source_tz="America/New_York", bar_end=False):
    """Generic CSV with a timestamp column (or date + time) and open/high/low/close/volume."""
    frames = []
    for p in paths:
        raw = pd.read_csv(p)
        raw.columns = [c.strip().lower() for c in raw.columns]
        if "date" in raw and "time" in raw:
            ts = pd.to_datetime(raw["date"].astype(str) + " " + raw["time"].astype(str))
        else:
            tcol = next(c for c in raw.columns if c in ("timestamp", "datetime", "time", "date", "ts_event"))
            ts = pd.to_datetime(raw[tcol])
        ts = pd.DatetimeIndex(ts)
        if ts.tz is None:
            ts = ts.tz_localize(source_tz, ambiguous="NaT", nonexistent="NaT")
        f = raw[COLS].set_index(ts)
        f = f[f.index.notna()]
        if bar_end:
            f.index = f.index - pd.Timedelta(minutes=_infer_minutes(f))
        f.index = f.index.tz_convert("America/New_York")
        frames.append(f)
    return clean(pd.concat(frames))


def download_yahoo(symbol, interval="5m"):
    """Free, but Yahoo only keeps ~60 days of 5m and ~7 days of 1m intraday futures data."""
    import yfinance as yf
    inst = get_instrument(symbol)
    period = {"1m": "7d", "2m": "60d", "5m": "60d", "15m": "60d", "30m": "60d", "60m": "730d"}[interval]
    df = yf.download(inst.yahoo, period=period, interval=interval, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError("Yahoo returned no data (network blocked or symbol unavailable).")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.DatetimeIndex(df.index).tz_convert("America/New_York")
    return clean(df), int(interval.rstrip("m"))


def download_databento(symbol, start, end, api_key=None):
    """5+ years of CME Globex 1-minute OHLCV for the volume-front-month contract.

    Needs `pip install databento` and an API key (env DATABENTO_API_KEY). Five years of MES
    1-minute bars costs a few dollars; new accounts receive free credit.
    """
    import databento as db
    inst = get_instrument(symbol)
    client = db.Historical(api_key or os.environ.get("DATABENTO_API_KEY"))
    cost = client.metadata.get_cost(dataset="GLBX.MDP3", symbols=[inst.databento], stype_in="continuous",
                                    schema="ohlcv-1m", start=start, end=end)
    print(f"Databento quoted cost for this request: ${cost:.2f}")
    data = client.timeseries.get_range(dataset="GLBX.MDP3", symbols=[inst.databento], stype_in="continuous",
                                       schema="ohlcv-1m", start=start, end=end)
    df = data.to_df()  # indexed by ts_event (bar start, UTC); prices already in points
    df.index = pd.DatetimeIndex(df.index).tz_convert("America/New_York")
    # the continuous symbol switches contracts at rolls; back-adjust using instrument_id
    contract = df["instrument_id"].astype(str).values
    if len(set(contract)) > 1:
        df = _adjust_by_instrument_id(df)
    return clean(df), 1


def _adjust_by_instrument_id(df):
    """For a vendor continuous series we only see one contract at a time; remove the price jump
    at each roll by aligning the first open of the new contract with the last close of the old."""
    df = df.copy()
    change = df["instrument_id"].ne(df["instrument_id"].shift()) & df["instrument_id"].shift().notna()
    gaps = (df["open"] - df["close"].shift()).where(change, 0.0)
    # shift everything BEFORE each roll by that roll's gap
    adj = gaps[::-1].cumsum()[::-1].shift(-1).fillna(0.0)
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c] + adj
    return df


def synthetic_bars(days=250, seed=1, start="2024-01-02", tick=0.25, edge=0.12):
    """Fake MES-like 1-minute Globex bars for testing the pipeline offline.

    A small edge is planted on purpose (momentum after the 15-minute opening range breaks), so a
    working learner should find it. Real markets offer no such guarantee - never judge the bot by
    its results on this data.
    """
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start, periods=days)
    price, frames = 4500.0, []
    minutes = np.arange(-360, 1020)              # 18:00 previous evening .. 16:59
    mod = (minutes % 1440)
    rth = (mod >= 570) & (mod < 960)
    sigma = np.where(rth, 0.55, 0.22) * (1 + 0.8 * np.exp(-((mod - 570) % 1440) / 30.0) * rth)
    vol_shape = np.where(rth, 1500, 150) * (1 + 2 * np.exp(-((mod - 570) % 1440) / 20.0) * rth)
    for d in sessions:
        n = len(minutes)
        shocks = rng.standard_t(4, n) * sigma * 0.7
        o = np.empty(n); c = np.empty(n)
        or_hi, or_lo, drift_left, drift = -np.inf, np.inf, 0, 0.0
        p = price
        for k in range(n):
            o[k] = p
            r = shocks[k]
            if drift_left > 0:
                r += drift
                drift_left -= 1
            p = p + r
            c[k] = p
            m = mod[k]
            if 570 <= m < 585:
                or_hi, or_lo = max(or_hi, o[k], p), min(or_lo, o[k], p)
            elif 585 <= m < 900 and drift_left == 0:
                if p > or_hi + 0.5:
                    drift_left, drift, or_hi = 20, edge * sigma[k], p + 2.0
                elif p < or_lo - 0.5:
                    drift_left, drift, or_lo = 20, -edge * sigma[k], p - 2.0
        price = p
        hi = np.maximum(o, c) + np.abs(rng.normal(0, 0.5, n)) * sigma
        lo = np.minimum(o, c) - np.abs(rng.normal(0, 0.5, n)) * sigma
        rnd = lambda x: np.round(x / tick) * tick
        idx = pd.DatetimeIndex(d + pd.to_timedelta(minutes, "min")).tz_localize("America/New_York",
                                                                                  ambiguous=True,
                                                                                  nonexistent="shift_forward")
        frames.append(pd.DataFrame({"open": rnd(o), "high": rnd(hi), "low": rnd(lo), "close": rnd(c),
                                    "volume": np.round(vol_shape * rng.lognormal(0, 0.4, n))}, index=idx))
    return clean(pd.concat(frames))
