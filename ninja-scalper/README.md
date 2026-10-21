# ninja-scalper: a learning scalper for NinjaTrader 8

A machine-learning scalping bot for CME micro index futures: **MES** (Micro E-mini S&P 500) by default, plus MNQ, M2K and MYM.

* **Learns from history.** It uses up to 5 years of 1-minute bars, downloaded over the internet or exported from NinjaTrader.
* **Explores timeframes.** It rebuilds the data as 1, 2, 3, 5, 10 and 15-minute bars and tests each one with several stop sizes and reward:risk targets of **2:1, 2.5:1 and 3:1**. It ranks every combination by out-of-sample results.
* **Trades through NinjaTrader.** A NinjaScript strategy sends each closed bar to the Python model. The model replies `long`, `short` or `none`, with a stop and a target in ticks. NinjaTrader then places the entry with an attached bracket.
* **Keeps learning.** `retrain` adds new data and refits the model. `learn` re-runs the whole timeframe search.

> **Risk warning.** This is a research tool, not a promise of profit. Most intraday strategies lose money after costs. Run the bot on **Sim101 / Playback** for weeks before you risk real money, and never trade a setup whose out-of-sample t-stat is below about 2.

---

## How it learns

```
5 years of 1-minute bars
   │  resample to 1 / 2 / 3 / 5 / 10 / 15 min
   ▼
features (≈40 per bar: momentum, EMA trend, VWAP distance, opening range, prior-day levels,
          RSI, Bollinger, candle shape, volume, volatility regime, time of day)
   │
   ▼
labels: "if I had entered on the next bar's open with stop = k×ATR and target = R×stop,
         would the target be hit before the stop?"   (separately for long and short)
   │
   ▼
gradient-boosted trees learn P(target before stop) for each side
   │
   ▼
walk-forward test: train on the past, pick a probability cut-off on the most recent part of
the training data, then trade the NEXT period the model has never seen; repeat over 5 windows
   │
   ▼
leaderboard of timeframe × stop size × reward:risk  →  the winner becomes the live model
```

With a 2:1 target the bot breaks even at a ~33% hit rate before costs. It only takes a trade when the model's probability is above the learned cut-off.

**The backtest assumes conservative fills, so its results are not flattered:**
- Entry is a market order at the next bar's open, plus 1 tick of slippage.
- If one bar touches both the stop and the target, the trade counts as a **loss**.
- A limit target counts as filled only when price trades **one tick through** it.
- A stop that price gaps through fills at the worse price, plus slippage.
- Commission is $0.62 per side.
- One position at a time. At most 8 trades a day, and trading stops for the day after a 3R loss.
- No trade is held across the session break. Every trade is closed by 15:55 ET.

**Reality check about 1-minute scalping:** on MES a 4-tick (1-point) stop risks $5. Commission ($1.24) plus two ticks of slippage ($2.50) cost about **0.75R per trade**. A 1-minute strategy needs a very strong edge just to break even. The explorer measures this for you, and it often shows that 3–15-minute bars scalp better than 1-minute bars.

---

## Setup (Windows, next to NinjaTrader 8)

### 1. Python

Install Python 3.11+ from python.org, then run:

```bat
cd ninja-scalper
python -m pip install -r requirements.txt
```

### 2. Get 5 years of data (choose one)

**A. Databento (recommended, over the internet).** It provides CME Globex 1-minute bars back to 2010. Five years of MES costs a few dollars, and new accounts get free credit. Create an account at databento.com, copy your API key, then run:

```bat
pip install databento
set DATABENTO_API_KEY=db-xxxxxxxx
python -m scalper download --source databento --years 5
```

The command prints the quoted cost before it downloads.

**B. NinjaTrader's own data (free with your NinjaTrader data feed).**
1. In the Control Center, open **Tools → Historical Data**.
2. Download MES 1-minute **Last** data for each quarterly contract (MES 03-21 … MES 12-26).
3. Use **Export** to save each contract as a text file.
4. Load all the files at once. The bot rolls between contracts on volume and back-adjusts the prices:

```bat
python -m scalper download --source ninjatrader --files "C:\exports\MES*.txt"
```

The timestamps follow NinjaTrader's time-zone setting. If yours is not Eastern, add `--source-tz America/Chicago` (or your zone). If the busiest bar of the day is not the 09:30 ET open, the loader prints a warning.

**C. Any 1-minute CSV** (FirstRate Data, Kibot, …), with columns `timestamp,open,high,low,close,volume`:

```bat
python -m scalper download --source csv --files mes_1min.csv --source-tz America/New_York
```

**D. Quick smoke test with no account.** Yahoo keeps only about 60 days of 5-minute data, which is too little to learn from but enough to check that everything works:

```bat
python -m scalper download --source yahoo --interval 5m
```

### 3. Learn

```bat
python -m scalper learn
```

This tests every timeframe, stop size and reward:risk combination walk-forward and trains the winner. On 5 years of data it takes about 10–30 minutes. The command writes:
- `reports/summary.md`: the best result per timeframe, all out-of-sample
- `reports/leaderboard.csv`: every combination tested
- `reports/best_oos_trades.csv`: every out-of-sample trade of the winning setup
- `models/MES_live.joblib` and `.json`: the live model and its settings

Useful variations:

```bat
python -m scalper explore --timeframes 1 3 5          :: just these timeframes
python -m scalper explore --quick                     :: one stop size, 2:1 only (fast)
python -m scalper train --timeframe 5 --stop-mult 1.0 --reward 2.5   :: pick a setup yourself
python -m scalper --instrument MNQ learn              :: Micro Nasdaq (download MNQ data first)
```

### 4. Connect NinjaTrader

1. Copy `ninjatrader\AiScalper.cs` to `Documents\NinjaTrader 8\bin\Custom\Strategies\`.
2. Open **New → NinjaScript Editor** and press **F5** to compile.
3. Start the bridge and leave it running. It prints the timeframe the chart must use:
   ```bat
   python -m scalper serve
   ```
4. Open a chart of the front MES contract, with **Minute** bars at the printed value. Set **Days to load** to 30 or more.
5. Add the **AiScalper** strategy and choose the **Sim101** account. Set the risk properties:
   - Contracts
   - Max trades per day
   - Daily loss limit ($)
   - Flatten time (HHmm, Eastern)
6. Enable the strategy. Historical bars warm up the model, and trading starts on live bars. Signals appear in NinjaTrader's Output window, and every decision is logged to `reports/live_signals.csv`.

To backtest the live model inside NinjaTrader, enable **Trade historical bars** in the Strategy Analyzer or Playback. Keep the server running while you do.

### 5. Keep it learning

```bat
python -m scalper retrain --databento     :: fetch bars since the last update, refit with the current settings
python -m scalper learn                   :: monthly: re-explore all timeframes on the larger history
```

If you use NinjaTrader exports, export the newest contract again and run `download --source ninjatrader` with that file. New bars are merged into the stored history. Then run `retrain`.

Restart `serve` after retraining. You can automate the weekly retrain with Windows Task Scheduler.

---

## Settings

The defaults are in `scalper/config.py`. To override any of them, write a JSON file and pass it with `--config`:

```json
{ "timeframes": [2, 3, 5], "reward_ratios": [2.0, 3.0], "trade_start": "09:45", "trade_end": "11:30",
  "max_trades_per_day": 4, "commission_per_side": 0.50 }
```

```bat
python -m scalper --config my.json learn
```

The reward:risk ratio is never allowed below 2:1. Both Python and NinjaTrader reject any bracket with a target smaller than 2× the stop.

## Project layout

| Path | What it is |
|---|---|
| `scalper/data.py` | Downloads and loads bars, stitches contract rolls, resamples timeframes, generates synthetic test data |
| `scalper/features.py` | Indicators and context features. Each bar uses only past bars, and the same code runs live |
| `scalper/labels.py` | Triple-barrier outcome of every possible scalp, with conservative fills |
| `scalper/backtest.py` | Trade simulation with risk limits and costs, plus metrics |
| `scalper/learn.py` | Walk-forward exploration, the leaderboard and final training |
| `scalper/server.py` | TCP bridge used by NinjaTrader |
| `ninjatrader/AiScalper.cs` | NinjaTrader 8 strategy |
| `tests/` | `python -m pytest tests`: fill rules, no look-ahead, live = backtest parity, learner sanity |
