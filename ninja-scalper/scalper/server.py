"""Live bridge: NinjaTrader (AiScalper.cs) sends every closed bar over a local TCP socket, this server
rebuilds the exact training features, asks the trained models for P(target before stop) and replies
with a trade instruction. Brackets (stop/target) and risk limits are then placed by NinjaTrader.

Protocol: one JSON object per line in each direction.
  NT -> {"type":"hello","instrument":"MES 12-26","period_min":1}
  NT -> {"type":"bar","time":"2026-10-20 09:31:00","period_min":1,"o":..,"h":..,"l":..,"c":..,"v":..,
         "predict":true}            (time = bar CLOSE time, US/Eastern)
  <- {"action":"long|short|none","prob":0.41,"stop_ticks":8,"target_ticks":16,"max_hold_bars":12,"reason":".."}
"""
import csv
import json
import os
import socketserver
import threading
from collections import deque

import joblib
import numpy as np
import pandas as pd

from .backtest import hhmm
from .features import build_features
from .instruments import get_instrument
from .labels import stop_target_ticks
from .learn import MODELS_DIR, REPORTS_DIR

BUFFER_BARS = 6000     # enough history for prior-session levels and fully converged averages
MIN_BARS = 400


class Brain:
    """Holds the model and the rolling bar buffer; decide() is pure logic so it can be tested
    without a socket."""

    def __init__(self, model_path):
        self.b = joblib.load(model_path)
        self.cfg = self.b["config"]
        self.inst = get_instrument(self.b["instrument"])
        self.minutes = self.b["timeframe_min"]
        self.bars = deque(maxlen=BUFFER_BARS)
        self.lock = threading.Lock()
        self.start, self.end = hhmm(self.cfg["trade_start"]), hhmm(self.cfg["trade_end"])
        os.makedirs(REPORTS_DIR, exist_ok=True)
        self.log_path = os.path.join(REPORTS_DIR, "live_signals.csv")

    def describe(self):
        return {"ok": True, "instrument": self.b["instrument"], "timeframe_min": self.minutes,
                "threshold": self.b["threshold"], "reward_risk": self.b["reward_risk"],
                "max_hold_bars": self.b["max_hold_bars"], "trained_through": self.b["trained_through"]}

    def add_bar(self, close_time, o, h, l, c, v):
        start = pd.Timestamp(close_time).tz_localize("America/New_York", ambiguous=True, nonexistent="shift_forward") \
            - pd.Timedelta(minutes=self.minutes)
        row = (start, float(o), float(h), float(l), float(c), float(v))
        if self.bars and self.bars[-1][0] == start:
            self.bars[-1] = row
        elif not self.bars or start > self.bars[-1][0]:
            self.bars.append(row)
        return start

    def frame(self):
        df = pd.DataFrame(list(self.bars), columns=["time", "open", "high", "low", "close", "volume"])
        return df.set_index(pd.DatetimeIndex(df.pop("time")))

    def decide(self, msg):
        none = {"action": "none", "prob": 0.0, "stop_ticks": 0, "target_ticks": 0,
                "max_hold_bars": self.b["max_hold_bars"]}
        if int(msg.get("period_min", self.minutes)) != self.minutes:
            return {**none, "reason": f"chart is {msg.get('period_min')}m but model is {self.minutes}m"}
        with self.lock:
            start = self.add_bar(msg["time"], msg["o"], msg["h"], msg["l"], msg["c"], msg.get("v", 0))
            if not msg.get("predict", False):
                return {**none, "reason": "warmup"}
            if len(self.bars) < MIN_BARS:
                return {**none, "reason": f"warming up ({len(self.bars)}/{MIN_BARS} bars)"}
            close_min = start.hour * 60 + start.minute + self.minutes
            if not (self.start <= close_min <= self.end):
                return {**none, "reason": "outside trading window"}
            X, aux = build_features(self.frame(), self.minutes)
        x = X[self.b["features"]].iloc[[-1]]
        p_long = float(self.b["model_long"].predict_proba(x)[0, 1]) if self.b["model_long"] else 0.0
        p_short = float(self.b["model_short"].predict_proba(x)[0, 1]) if self.b["model_short"] else 0.0
        stop, target = stop_target_ticks([aux["atr"].iloc[-1]], self.inst.tick_size, self.b["stop_atr_mult"],
                                          self.b["reward_risk"], self.cfg["min_stop_ticks"], self.cfg["max_stop_ticks"])
        side = "long" if p_long >= p_short else "short"
        prob = max(p_long, p_short)
        action = side if prob >= self.b["threshold"] else "none"
        out = {"action": action, "prob": round(prob, 4), "stop_ticks": int(stop[0]), "target_ticks": int(target[0]),
               "max_hold_bars": self.b["max_hold_bars"], "reason": f"pL={p_long:.3f} pS={p_short:.3f}"}
        self._log(start, msg, p_long, p_short, out)
        return out

    def _log(self, start, msg, p_long, p_short, out):
        new = not os.path.exists(self.log_path)
        with open(self.log_path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["bar_start", "close", "p_long", "p_short", "action", "stop_ticks", "target_ticks"])
            w.writerow([start.isoformat(), msg["c"], f"{p_long:.4f}", f"{p_short:.4f}", out["action"],
                        out["stop_ticks"], out["target_ticks"]])


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        brain = self.server.brain
        print(f"NinjaTrader connected from {self.client_address}")
        for raw in self.rfile:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                if msg.get("type") == "hello":
                    reply = brain.describe()
                    if int(msg.get("period_min", brain.minutes)) != brain.minutes:
                        reply["warning"] = f"Set the chart to {brain.minutes}-minute bars"
                    brain.bars.clear()
                else:
                    reply = brain.decide(msg)
                    if reply["action"] != "none":
                        print(f"{msg['time']}  {reply['action'].upper():5}  p={reply['prob']:.3f}  "
                              f"stop {reply['stop_ticks']}t  target {reply['target_ticks']}t")
            except Exception as e:  # never kill the connection over one bad message
                reply = {"action": "none", "error": str(e)}
            self.wfile.write((json.dumps(reply) + "\n").encode("utf-8"))
            self.wfile.flush()
        print("NinjaTrader disconnected")


class Server(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(model_path=None, host="127.0.0.1", port=5555, instrument="MES"):
    model_path = model_path or os.path.join(MODELS_DIR, f"{instrument}_live.joblib")
    brain = Brain(model_path)
    info = brain.describe()
    print(f"Loaded {model_path}: {info}")
    with Server((host, port), Handler) as srv:
        srv.brain = brain
        print(f"Listening on {host}:{port} - start the AiScalper strategy in NinjaTrader "
              f"on a {brain.minutes}-minute {info['instrument']} chart.")
        srv.serve_forever()
