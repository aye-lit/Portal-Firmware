import json

import numpy as np
import pandas as pd
import pytest

from scalper import data
from scalper.config import Config
from scalper.features import build_features
from scalper.instruments import get_instrument
from scalper.labels import label

TICK = 0.25


def bars_from(rows, start="2025-03-04 10:00"):
    idx = pd.date_range(start, periods=len(rows), freq="1min", tz="America/New_York")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx).assign(volume=100.0)


def run_label(df, side, atr=1.0, reward=2.0, max_hold=10):
    aux = pd.DataFrame({"atr": atr, "sdate": data.session_date(df.index),
                        "mod": df.index.hour * 60 + df.index.minute}, index=df.index)
    return label(df, aux, side, tick=TICK, stop_mult=1.0, reward=reward, max_hold=max_hold, minutes=1,
                 flatten_min=15 * 60 + 55, min_ticks=4, max_ticks=40, slip_ticks=1)


def test_long_target_hit():
    # signal on bar 0; entry bar 1 open 100 (+1 tick slip = 100.25); stop 4 ticks, target 8 ticks -> 102.25
    df = bars_from([(100, 100, 100, 100), (100, 100.5, 99.75, 100.25), (100.5, 102.5, 100.5, 102.5),
                    (102.5, 102.5, 102.5, 102.5)])
    out = run_label(df, +1)
    assert out.valid[0] and out.win[0]
    assert out.exit_idx[0] == 2
    assert out.pnl_pts[0] == pytest.approx(2.0)          # 8 ticks


def test_target_touch_is_not_a_fill():
    df = bars_from([(100, 100, 100, 100), (100, 100.5, 99.75, 100.25), (100.5, 102.25, 100.5, 102),
                    (102, 102, 102, 102)])
    out = run_label(df, +1, max_hold=2)
    assert not out.win[0]


def test_stop_wins_ties_and_slips():
    # bar 2 touches both stop (99.25) and target -> counted as a loss, filled 1 tick worse
    df = bars_from([(100, 100, 100, 100), (100, 100.25, 100, 100.25), (100, 103, 99, 101), (101, 101, 101, 101)])
    out = run_label(df, +1)
    assert not out.win[0]
    assert out.pnl_pts[0] == pytest.approx(99.0 - 100.25)


def test_short_side():
    df = bars_from([(100, 100, 100, 100), (100, 100.25, 99.75, 99.75), (99.5, 99.5, 97.5, 97.5), (97.5,) * 4])
    out = run_label(df, -1)
    assert out.win[0] and out.pnl_pts[0] == pytest.approx(2.0)


def test_trades_never_cross_the_session_break():
    # 17:58 and 17:59 belong to one CME session, 18:00 starts the next one
    df = bars_from([(100, 100, 100, 100), (100, 100, 100, 100), (100, 110, 100, 110), (110,) * 4],
                   start="2025-03-04 17:58")
    aux = pd.DataFrame({"atr": 1.0, "sdate": data.session_date(df.index),
                        "mod": df.index.hour * 60 + df.index.minute}, index=df.index)
    out = label(df, aux, +1, tick=TICK, stop_mult=1.0, reward=2.0, max_hold=10, minutes=1, flatten_min=24 * 60,
                min_ticks=4, max_ticks=40, slip_ticks=1)
    assert out.valid[0] and not out.win[0] and out.exit_idx[0] == 1   # timed out at the session end
    assert not out.valid[1]                                           # entry would be in the next session


def test_features_have_no_lookahead():
    base = data.synthetic_bars(days=6, seed=5)
    f1, _ = build_features(base, 1)
    tampered = base.copy()
    tampered.iloc[-500:, :4] *= 1.05
    f2, _ = build_features(tampered, 1)
    cut = len(base) - 500
    pd.testing.assert_frame_equal(f1.iloc[:cut], f2.iloc[:cut])


def test_server_matches_training_features(tmp_path, monkeypatch):
    """The live path (bar-by-bar through the server) must produce the same probability as the batch
    path used in backtests."""
    import joblib
    from scalper import learn, server
    monkeypatch.setattr(learn, "MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(server, "REPORTS_DIR", str(tmp_path))
    cfg = Config(n_folds=2, prob_thresholds=[0.0], min_calib_trades=1)
    base = data.synthetic_bars(days=40, seed=2, edge=0.3)
    path = learn.train(base, 1, cfg, 5, 1.0, 2.0, log=lambda *a: None)
    bundle = joblib.load(path)
    bars = data.resample(base, 5)

    # predict on a bar inside the trading window
    t = bars.index[(bars.index.hour == 11) & (bars.index.minute == 0)][-1]
    window = bars.loc[:t]
    brain = server.Brain(path)
    for tt, r in window.iloc[-server.BUFFER_BARS:].iterrows():
        msg = {"time": str((tt + pd.Timedelta(minutes=5)).tz_localize(None)), "period_min": 5, "o": r.open,
               "h": r.high, "l": r.low, "c": r.close, "v": r.volume, "predict": tt == t}
        reply = brain.decide(msg)
    X, _ = build_features(bars, 5)
    x = X[bundle["features"]].loc[[t]]
    p_long = bundle["model_long"].predict_proba(x)[0, 1]
    p_short = bundle["model_short"].predict_proba(x)[0, 1]
    assert reply["prob"] == pytest.approx(max(p_long, p_short), abs=2e-3)
    json.dumps(reply)


def test_learner_finds_a_real_edge_and_rejects_noise():
    from scalper.learn import explore
    cfg = Config(timeframes=[5], stop_atr_mults=[1.0], reward_ratios=[2.0], n_folds=3)
    quiet = lambda *a: None
    edge = explore(data.synthetic_bars(days=200, edge=0.35, seed=3), 1, cfg, log=quiet).iloc[0]
    noise = explore(data.synthetic_bars(days=200, edge=0.0, seed=3), 1, cfg, log=quiet).iloc[0]
    assert edge.avg_r > 0.3 and edge.t_stat > 3
    assert noise.avg_r < 0.1


def test_back_adjust_removes_roll_gap():
    idx = pd.date_range("2025-03-10 09:30", periods=3, freq="1D", tz="America/New_York")
    old = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": [10, 10, 1]}, index=idx)
    new = pd.DataFrame({"open": 120.0, "high": 120.0, "low": 120.0, "close": 120.0, "volume": [1, 1, 10]}, index=idx)
    both = pd.concat([old, new])
    out = data.back_adjust(both, ["MES 03-25"] * 3 + ["MES 06-25"] * 3)
    assert list(out["close"]) == [120.0, 120.0, 120.0]


def test_instrument_specs():
    mes = get_instrument("MES 12-26")
    assert mes.tick_value == pytest.approx(1.25)


def test_account_sizes_each_trade_to_one_percent_and_compounds():
    from scalper.backtest import account
    cfg = Config(starting_capital=2000, risk_per_trade_pct=1.0)
    # risk of 1 contract is $10 -> 1% of $2,000 = $20 -> 2 contracts; a +$500 trade then lifts the size
    trades = pd.DataFrame({"net_usd": [250.0, -10.0], "risk_usd": [10.0, 10.0], "session": [1, 2]})
    a = account(trades, cfg, n_days=2)
    assert a["taken"] == 2
    assert a["end_usd"] == pytest.approx(2000 + 2 * 250 - 2 * 10)   # 1% of $2,500 = $25 -> still 2 contracts


def test_account_skips_trades_it_cannot_size():
    from scalper.backtest import account
    cfg = Config(starting_capital=200, risk_per_trade_pct=1.0)   # 1% = $2, less than any MES stop
    trades = pd.DataFrame({"net_usd": [50.0, -10.0], "risk_usd": [5.0, 10.0], "session": [1, 2]})
    a = account(trades, cfg, n_days=2)
    assert a["taken"] == 0 and a["end_usd"] == 200


def test_clean_drops_weekend_prints():
    idx = pd.DatetimeIndex(["2025-09-19 15:00", "2025-09-20 21:39", "2025-09-21 17:16", "2025-09-21 18:01"],
                           tz="America/New_York")   # Fri, Sat, Sun before the open, Sun after the 18:00 open
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)
    assert list(data.clean(df).index) == [idx[0], idx[3]]
