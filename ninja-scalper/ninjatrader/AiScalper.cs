// AiScalper - NinjaTrader 8 strategy that trades the model trained by ninja-scalper.
//
// Install: copy this file to  Documents\NinjaTrader 8\bin\Custom\Strategies\AiScalper.cs, open
// New > NinjaScript Editor and press F5 to compile. Then start `python -m scalper serve` and add the
// strategy to a chart whose bar period matches the trained model (the server prints it, e.g. "5-minute MES").
//
// Each closed bar is sent to the Python server; when flat and inside the risk limits the server may answer
// "long" or "short" with a stop and target in ticks (target >= 2x stop). This strategy places the entry with
// an attached stop-loss and profit-target bracket, closes trades that exceed the model's max hold time, and
// flattens before the session close.
#region Using declarations
using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Globalization;
using System.IO;
using System.Net.Sockets;
using System.Text;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class AiScalper : Strategy
    {
        private TcpClient client;
        private StreamReader reader;
        private StreamWriter writer;
        private DateTime lastConnectAttempt = DateTime.MinValue;
        private bool needsReplay;
        private int tradesToday;
        private double sessionStartPnl;
        private int maxHoldBars = 12;
        private TimeZoneInfo eastern;
        private const int ReplayBars = 6000;
        private const string LongSignal = "AIL";
        private const string ShortSignal = "AIS";

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name = "AiScalper";
                Description = "Machine-learning scalper for micro index futures (needs `python -m scalper serve`).";
                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 60;
                StartBehavior = StartBehavior.WaitUntilFlat;
                BarsRequiredToTrade = 20;
                TraceOrders = false;

                Host = "127.0.0.1";
                Port = 5555;
                Contracts = 1;
                MaxTradesPerDay = 8;
                DailyLossLimit = 30;      // sized for a $200 example account
                MaxRiskPerTrade = 15;
                FlattenTimeEt = 1555;
                TradeHistorical = false;
            }
            else if (State == State.DataLoaded)
            {
                eastern = TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time");
                if (BarsPeriod.BarsPeriodType != BarsPeriodType.Minute)
                    Print("AiScalper: use a Minute chart; the model was trained on minute bars.");
                Connect();
            }
            else if (State == State.Terminated)
            {
                Disconnect();
            }
        }

        protected override void OnBarUpdate()
        {
            if (BarsInProgress != 0 || CurrentBar < 1)
                return;

            if (Bars.IsFirstBarOfSession)
            {
                tradesToday = 0;
                sessionStartPnl = SystemPerformance.AllTrades.TradesPerformance.Currency.CumProfit;
            }

            DateTime closeEt = ToEastern(Time[0]);
            int hhmm = closeEt.Hour * 100 + closeEt.Minute;

            // ---- manage an open trade: time stop and end-of-day flatten (brackets handle stop/target)
            if (Position.MarketPosition != MarketPosition.Flat)
            {
                // entry filled at the open of the bar after the signal; the model's trades last at most
                // maxHoldBars bars including the entry bar
                bool timeUp = BarsSinceEntryExecution() >= maxHoldBars - 1;
                if (timeUp || hhmm >= FlattenTimeEt)
                {
                    if (Position.MarketPosition == MarketPosition.Long)
                        ExitLong(timeUp ? "TimeExit" : "Flatten", LongSignal);
                    else
                        ExitShort(timeUp ? "TimeExit" : "Flatten", ShortSignal);
                }
            }

            if (!EnsureConnected())
                return;
            if (needsReplay)
                ReplayHistory();

            bool flat = Position.MarketPosition == MarketPosition.Flat;
            bool canTrade = State == State.Realtime || TradeHistorical;
            double dayPnl = SystemPerformance.AllTrades.TradesPerformance.Currency.CumProfit - sessionStartPnl;
            bool allowed = flat && canTrade && CurrentBar >= BarsRequiredToTrade
                           && tradesToday < MaxTradesPerDay && dayPnl > -DailyLossLimit && hhmm < FlattenTimeEt;

            string reply = Exchange(BarJson(0, allowed));
            if (reply == null || !allowed)
                return;

            string action = JsonValue(reply, "action");
            if (action != "long" && action != "short")
                return;

            int stopTicks = ParseInt(JsonValue(reply, "stop_ticks"));
            int targetTicks = ParseInt(JsonValue(reply, "target_ticks"));
            int hold = ParseInt(JsonValue(reply, "max_hold_bars"));
            if (stopTicks <= 0 || targetTicks < 2 * stopTicks)
            {
                Print("AiScalper: rejected signal with bad bracket " + reply);
                return;
            }
            double riskUsd = stopTicks * TickSize * Instrument.MasterInstrument.PointValue * Contracts;
            if (riskUsd > MaxRiskPerTrade)
            {
                Print(string.Format("AiScalper: skipped {0}, risk ${1:0.00} > ${2:0.00}", action, riskUsd, MaxRiskPerTrade));
                return;
            }
            if (hold > 0)
                maxHoldBars = hold;

            if (action == "long")
            {
                SetStopLoss(LongSignal, CalculationMode.Ticks, stopTicks, false);
                SetProfitTarget(LongSignal, CalculationMode.Ticks, targetTicks);
                EnterLong(Contracts, LongSignal);
            }
            else
            {
                SetStopLoss(ShortSignal, CalculationMode.Ticks, stopTicks, false);
                SetProfitTarget(ShortSignal, CalculationMode.Ticks, targetTicks);
                EnterShort(Contracts, ShortSignal);
            }
            tradesToday++;
            Print(string.Format("AiScalper {0} {1} p={2} stop={3}t target={4}t", Time[0], action.ToUpper(),
                JsonValue(reply, "prob"), stopTicks, targetTicks));
        }

        // ------------------------------------------------------------------ bridge

        private DateTime ToEastern(DateTime t)
        {
            return TimeZoneInfo.ConvertTime(DateTime.SpecifyKind(t, DateTimeKind.Unspecified),
                Core.Globals.GeneralOptions.TimeZoneInfo, eastern);
        }

        private string BarJson(int barsAgo, bool predict)
        {
            CultureInfo ci = CultureInfo.InvariantCulture;
            return string.Format(ci,
                "{{\"type\":\"bar\",\"time\":\"{0:yyyy-MM-dd HH:mm:ss}\",\"period_min\":{1},\"o\":{2},\"h\":{3},\"l\":{4},\"c\":{5},\"v\":{6},\"predict\":{7}}}",
                ToEastern(Time[barsAgo]), BarsPeriod.Value, Open[barsAgo], High[barsAgo], Low[barsAgo], Close[barsAgo],
                Volume[barsAgo], predict ? "true" : "false");
        }

        private void Connect()
        {
            lastConnectAttempt = DateTime.Now;
            try
            {
                client = new TcpClient();
                client.Connect(Host, Port);
                client.NoDelay = true;
                client.ReceiveTimeout = 5000;
                client.SendTimeout = 5000;
                NetworkStream ns = client.GetStream();
                reader = new StreamReader(ns, new UTF8Encoding(false));
                writer = new StreamWriter(ns, new UTF8Encoding(false)) { AutoFlush = true, NewLine = "\n" };
                string hello = Exchange(string.Format(CultureInfo.InvariantCulture,
                    "{{\"type\":\"hello\",\"instrument\":\"{0}\",\"period_min\":{1}}}", Instrument.FullName, BarsPeriod.Value));
                Print("AiScalper connected: " + hello);
                needsReplay = CurrentBar > 0;   // server starts with an empty buffer after hello
            }
            catch (Exception e)
            {
                Print("AiScalper: cannot reach the Python server at " + Host + ":" + Port + " (" + e.Message +
                      "). Run `python -m scalper serve`.");
                Disconnect();
            }
        }

        private bool EnsureConnected()
        {
            if (client != null && client.Connected && writer != null)
                return true;
            if ((DateTime.Now - lastConnectAttempt).TotalSeconds < 10)
                return false;
            Connect();
            return client != null && client.Connected;
        }

        private void ReplayHistory()
        {
            needsReplay = false;
            for (int i = Math.Min(CurrentBar, ReplayBars); i >= 1; i--)
                if (Exchange(BarJson(i, false)) == null)
                    return;
        }

        private string Exchange(string line)
        {
            try
            {
                writer.WriteLine(line);
                string reply = reader.ReadLine();
                if (reply == null)
                    throw new IOException("server closed the connection");
                string err = JsonValue(reply, "error");
                if (!string.IsNullOrEmpty(err))
                    Print("AiScalper server error: " + err);
                return reply;
            }
            catch (Exception e)
            {
                Print("AiScalper: lost connection (" + e.Message + ")");
                Disconnect();
                return null;
            }
        }

        private void Disconnect()
        {
            try { if (writer != null) writer.Dispose(); } catch { }
            try { if (reader != null) reader.Dispose(); } catch { }
            try { if (client != null) client.Close(); } catch { }
            writer = null;
            reader = null;
            client = null;
        }

        // minimal reader for the flat JSON objects the server sends
        private static string JsonValue(string json, string key)
        {
            string token = "\"" + key + "\":";
            int i = json.IndexOf(token, StringComparison.Ordinal);
            if (i < 0)
                return null;
            i += token.Length;
            while (i < json.Length && json[i] == ' ')
                i++;
            if (i < json.Length && json[i] == '"')
            {
                int end = json.IndexOf('"', i + 1);
                return end < 0 ? null : json.Substring(i + 1, end - i - 1);
            }
            int stop = i;
            while (stop < json.Length && json[stop] != ',' && json[stop] != '}')
                stop++;
            return json.Substring(i, stop - i).Trim();
        }

        private static int ParseInt(string s)
        {
            double d;
            return double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out d) ? (int)Math.Round(d) : 0;
        }

        #region Properties
        [NinjaScriptProperty]
        [Display(Name = "Server host", GroupName = "Bridge", Order = 1)]
        public string Host { get; set; }

        [NinjaScriptProperty]
        [Range(1, 65535)]
        [Display(Name = "Server port", GroupName = "Bridge", Order = 2)]
        public int Port { get; set; }

        [NinjaScriptProperty]
        [Range(1, int.MaxValue)]
        [Display(Name = "Contracts", GroupName = "Risk", Order = 1)]
        public int Contracts { get; set; }

        [NinjaScriptProperty]
        [Range(1, int.MaxValue)]
        [Display(Name = "Max trades per day", GroupName = "Risk", Order = 2)]
        public int MaxTradesPerDay { get; set; }

        [NinjaScriptProperty]
        [Range(0, double.MaxValue)]
        [Display(Name = "Daily loss limit ($)", GroupName = "Risk", Order = 3)]
        public double DailyLossLimit { get; set; }

        [NinjaScriptProperty]
        [Range(0, double.MaxValue)]
        [Display(Name = "Max risk per trade ($)", GroupName = "Risk", Order = 5)]
        public double MaxRiskPerTrade { get; set; }

        [NinjaScriptProperty]
        [Range(0, 2359)]
        [Display(Name = "Flatten time (HHmm, Eastern)", GroupName = "Risk", Order = 4)]
        public int FlattenTimeEt { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Trade historical bars (Strategy Analyzer / playback)", GroupName = "Bridge", Order = 3)]
        public bool TradeHistorical { get; set; }
        #endregion
    }
}
