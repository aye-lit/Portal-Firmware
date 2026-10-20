"""All tunable settings in one place. Override any field from a JSON file with --config."""
import json
from dataclasses import dataclass, field, asdict, fields
from typing import List


@dataclass
class Config:
    instrument: str = "MES"
    timezone: str = "America/New_York"   # every timestamp is converted to exchange-local Eastern time

    # --- what the learner explores -------------------------------------------------
    timeframes: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 15])   # minutes
    stop_atr_mults: List[float] = field(default_factory=lambda: [0.75, 1.0, 1.5])  # stop = k * ATR(14)
    reward_ratios: List[float] = field(default_factory=lambda: [2.0, 2.5, 3.0])    # target = R * stop (>= 2:1)

    # --- trade mechanics (identical in backtest and live) --------------------------
    min_stop_ticks: int = 4              # never risk less than 1 MES point
    max_stop_ticks: int = 40             # never risk more than 10 MES points
    max_hold_minutes: int = 60           # scalps that have not resolved are closed at market
    trade_start: str = "09:35"           # Eastern; signals before this are ignored
    trade_end: str = "15:30"             # Eastern; no new entries after this
    flatten_time: str = "15:55"          # Eastern; open trades are closed here
    max_trades_per_day: int = 8
    daily_loss_limit_r: float = 3.0      # stop for the day after losing this many R

    # --- costs ---------------------------------------------------------------------
    commission_per_side: float = 0.62    # dollars per contract per side (check your broker)
    slippage_ticks: int = 1              # on market entries and stop exits

    # --- learning / validation ------------------------------------------------------
    initial_train_frac: float = 0.5      # first half of the history is the first training set
    n_folds: int = 5                     # the rest is split into this many walk-forward test windows
    calib_frac: float = 0.25             # tail of each training window used to pick the probability threshold
    prob_thresholds: List[float] = field(default_factory=lambda: [0.30, 0.33, 0.36, 0.40, 0.44, 0.48, 0.52, 0.56, 0.60])
    min_calib_trades: int = 20
    max_train_rows: int = 400_000        # subsample very large 1-minute training sets
    random_state: int = 7

    @classmethod
    def load(cls, path=None, **overrides):
        data = {}
        if path:
            with open(path) as f:
                data = json.load(f)
        data.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self):
        return asdict(self)
