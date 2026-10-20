"""Contract specs for the CME micro equity-index futures."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    symbol: str          # NinjaTrader master instrument name
    tick_size: float     # minimum price increment
    point_value: float   # dollars per 1.0 point
    yahoo: str           # Yahoo Finance continuous symbol
    databento: str       # Databento continuous symbol (front month by volume)

    @property
    def tick_value(self) -> float:
        return self.tick_size * self.point_value


INSTRUMENTS = {
    "MES": Instrument("MES", 0.25, 5.0, "MES=F", "MES.v.0"),   # Micro E-mini S&P 500
    "MNQ": Instrument("MNQ", 0.25, 2.0, "MNQ=F", "MNQ.v.0"),   # Micro E-mini Nasdaq-100
    "M2K": Instrument("M2K", 0.10, 5.0, "M2K=F", "M2K.v.0"),   # Micro E-mini Russell 2000
    "MYM": Instrument("MYM", 1.00, 0.5, "MYM=F", "MYM.v.0"),   # Micro E-mini Dow
}


def get_instrument(symbol: str) -> Instrument:
    key = symbol.upper().split()[0]  # accept "MES 12-26"
    if key not in INSTRUMENTS:
        raise KeyError(f"Unknown instrument {symbol!r}; known: {', '.join(INSTRUMENTS)}")
    return INSTRUMENTS[key]
