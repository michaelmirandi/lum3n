#!/usr/bin/env python3
import asyncio
from datetime import datetime, timezone, date
from ib_insync import IB, Stock, Option
import math
 
# -------- Config --------
IB_HOST = '127.0.0.1'
IB_PORT = 7496       # 7497 = paper, 7496 = live
CLIENT_ID = 101
SYMBOLS = ('QQQ', )  # QQQ only!
BAR_SEC = 5          # IB minimum is 5 seconds
USE_RTH = False
SHOW_OPTIONS = True  # Display 0DTE options
STRIKE_RANGE = 5     # Show +/- 5 strikes from ATM
# ------------------------
 
# ANSI colors
RED   = "\033[31m"
GREEN = "\033[32m"
YELL  = "\033[33m"
BLUE  = "\033[34m"
MAG   = "\033[35m"
CYAN  = "\033[36m"
RESET = "\033[0m"
 
class CandleAggregator:
    """
    Logs 5s bars and aggregates -> 1m candles -> 5m candles -> 15m candles.
    """
    def __init__(self):
        # rolling state for current candles per symbol
        self.state_1m = {}
        self.state_5m = {}
        self.state_15m = {}
        self.last_price = {}  # Track last price for options chain
 
    def on_bar_5s(self, sym, ts_utc, o, h, l, c, v):
        # Track last price for options
        self.last_price[sym] = c
        
        # LOG EVERY 5s BAR
        ts_str = ts_utc.strftime("%Y-%m-%d %H:%M:%S")
        col = GREEN if c > o else RED if c < o else YELL
        print(f"5s  | {ts_str} | {sym:<5} "
              f"O:{o:8.2f} H:{h:8.2f} L:{l:8.2f} "
              f"C:{col}{c:8.2f}{RESET} V:{v:10.0f}")
        
        # Round down to minute for 1m aggregation
        min_key = ts_utc.replace(second=0, microsecond=0)
        
        # Check if we need to close out 5m candle
        if min_key.minute % 5 == 0 and min_key.second == 0:
            st5 = self.state_5m.get(sym)
            if st5 and st5["minute"] != min_key:
                self._log(sym, st5, "5m")
                # push into 15m aggregator
                self._aggregate_15m(sym, st5)
                
        # Check if we need to close out 15m candle  
        if min_key.minute % 15 == 0 and min_key.second == 0:
            st15 = self.state_15m.get(sym)
            if st15 and st15["minute"] != min_key:
                self._log(sym, st15, "15m")
        
        # ---- 1 Minute candle aggregation ----
        st = self.state_1m.get(sym)
        if st is None or st["minute"] != min_key:
            if st is not None:
                self._log(sym, st, "1m")
                # push into 5m aggregator
                self._aggregate_5m(sym, st)
            self.state_1m[sym] = {"minute": min_key, "o": o, "h": h, "l": l, "c": c, "v": v}
        else:
            st["c"] = c
            st["h"] = max(st["h"], h)
            st["l"] = min(st["l"], l)
            st["v"] += v
 
    def _aggregate_5m(self, sym, one_min_candle):
        min_key = one_min_candle["minute"]
        # floor minute to nearest multiple of 5
        floored_minute = min_key.replace(minute=(min_key.minute // 5) * 5, second=0, microsecond=0)
        st5 = self.state_5m.get(sym)
        if st5 is None or st5["minute"] != floored_minute:
            # Don't log here - we'll log when the period actually ends
            self.state_5m[sym] = {
                "minute": floored_minute,
                "o": one_min_candle["o"],
                "h": one_min_candle["h"],
                "l": one_min_candle["l"],
                "c": one_min_candle["c"],
                "v": one_min_candle["v"],
            }
        else:
            st5["c"] = one_min_candle["c"]
            st5["h"] = max(st5["h"], one_min_candle["h"])
            st5["l"] = min(st5["l"], one_min_candle["l"])
            st5["v"] += one_min_candle["v"]
    
    def _aggregate_15m(self, sym, five_min_candle):
        min_key = five_min_candle["minute"]
        # floor minute to nearest multiple of 15
        floored_minute = min_key.replace(minute=(min_key.minute // 15) * 15, second=0, microsecond=0)
        st15 = self.state_15m.get(sym)
        if st15 is None or st15["minute"] != floored_minute:
            # Don't log here - we'll log when the period actually ends
            self.state_15m[sym] = {
                "minute": floored_minute,
                "o": five_min_candle["o"],
                "h": five_min_candle["h"],
                "l": five_min_candle["l"],
                "c": five_min_candle["c"],
                "v": five_min_candle["v"],
            }
        else:
            st15["c"] = five_min_candle["c"]
            st15["h"] = max(st15["h"], five_min_candle["h"])
            st15["l"] = min(st15["l"], five_min_candle["l"])
            st15["v"] += five_min_candle["v"]
 
    def _log(self, sym, st, tf):
        ts = st["minute"].strftime("%Y-%m-%d %H:%M")
        o,h,l,c,v = st["o"], st["h"], st["l"], st["c"], st["v"]
        # color close vs open
        if c > o: col = GREEN
        elif c < o: col = RED
        else: col = YELL
        print(f"{tf:<3} | {ts} | {sym:<5} "
              f"O:{o:8.2f} H:{h:8.2f} L:{l:8.2f} "
              f"C:{col}{c:8.2f}{RESET} V:{v:10.0f}")
 
    def flush(self):
        # Flush all remaining candles
        for sym, st in list(self.state_1m.items()):
            self._log(sym, st, "1m")
            self._aggregate_5m(sym, st)
        for sym, st5 in list(self.state_5m.items()):
            self._log(sym, st5, "5m")
            self._aggregate_15m(sym, st5)
        for sym, st15 in list(self.state_15m.items()):
            self._log(sym, st15, "15m")
        self.state_1m.clear()
        self.state_5m.clear()
        self.state_15m.clear()
 
class OneMinuteStreamer:
    def __init__(self):
        self.ib = IB()
        self.contracts = {}
        self.agg = CandleAggregator()
        self.options_data = {}
        self.last_options_update = None
 
    async def connect(self):
        await self.ib.connectAsync(IB_HOST, IB_PORT, clientId=CLIENT_ID)
        print("Connected to IBKR")
        self.ib.barUpdateEvent += self._on_bar
        # qualify and subscribe
        for sym in SYMBOLS:
            print(f"Subscribing to {sym}")
            [qc] = await self.ib.qualifyContractsAsync(Stock(sym, 'ARCA', 'USD'))
            self.contracts[sym] = qc
            self.ib.reqRealTimeBars(qc, BAR_SEC, 'TRADES', useRTH=USE_RTH, realTimeBarsOptions=[])
            
            # Get 0DTE options chain on startup
            if SHOW_OPTIONS:
                await self.get_0dte_options(sym)
 
    async def get_0dte_options(self, sym):
        """Fetch and display 0DTE ITM options only - FAST"""
        try:
            # Get today's date in YYYYMMDD format
            today = date.today().strftime('%Y%m%d')
            
            # Get current price from last bar if available, else fetch it
            if sym in self.agg.last_price:
                current_price = self.agg.last_price[sym]
            else:
                ticker = self.ib.reqMktData(self.contracts[sym])
                await asyncio.sleep(0.5)
                current_price = ticker.last or ticker.close or 572
                self.ib.cancelMktData(self.contracts[sym])
            
            # Calculate ATM strike
            atm_strike = round(current_price)
            
            # ONLY ITM strikes - no ATM, no OTM
            call_strikes = [s for s in range(atm_strike - 5, atm_strike) if s < current_price]  # ITM calls only
            put_strikes = [s for s in range(atm_strike + 1, atm_strike + 6) if s > current_price]  # ITM puts only
            
            print(f"\n{CYAN}═══ ITM 0DTE OPTIONS - {sym} @ ${current_price:.2f} ═══{RESET}")
            
            if call_strikes:
                print(f"{GREEN}ITM CALLS:{RESET}")
                print(f"{'Strike':<8} {'Bid':<8} {'Ask':<8} {'Mid':<8} {'Intrinsic':<10}")
                print("─" * 45)
                
                # Fetch only ITM calls
                for strike in call_strikes:
                    try:
                        call = Option(sym, today, strike, 'C', 'SMART')
                        [call_contract] = await self.ib.qualifyContractsAsync(call)
                        
                        # Snapshot=True auto-cancels, no need to manually cancel!
                        ticker = self.ib.reqMktData(call_contract, snapshot=True)
                        await asyncio.sleep(0.2)  # Faster wait
                        
                        bid = ticker.bid if ticker.bid and ticker.bid > 0 else 0
                        ask = ticker.ask if ticker.ask and ticker.ask > 0 else 0
                        mid = (bid + ask) / 2 if bid and ask else 0
                        intrinsic = current_price - strike
                        
                        # Highlight deep ITM
                        prefix = f"{MAG}*{RESET}" if intrinsic > 3 else " "
                        
                        print(f"{prefix}{strike:<7} {bid:>7.2f} {ask:>7.2f} {mid:>7.2f} {intrinsic:>9.2f}")
                        
                    except:
                        continue
            
            if put_strikes:
                print(f"\n{RED}ITM PUTS:{RESET}")
                print(f"{'Strike':<8} {'Bid':<8} {'Ask':<8} {'Mid':<8} {'Intrinsic':<10}")
                print("─" * 45)
                
                # Fetch only ITM puts
                for strike in put_strikes:
                    try:
                        put = Option(sym, today, strike, 'P', 'SMART')
                        [put_contract] = await self.ib.qualifyContractsAsync(put)
                        
                        ticker = self.ib.reqMktData(put_contract, snapshot=True)
                        await asyncio.sleep(0.2)  # Faster wait
                        
                        bid = ticker.bid if ticker.bid and ticker.bid > 0 else 0
                        ask = ticker.ask if ticker.ask and ticker.ask > 0 else 0
                        mid = (bid + ask) / 2 if bid and ask else 0
                        intrinsic = strike - current_price
                        
                        # Highlight deep ITM
                        prefix = f"{MAG}*{RESET}" if intrinsic > 3 else " "
                        
                        print(f"{prefix}{strike:<7} {bid:>7.2f} {ask:>7.2f} {mid:>7.2f} {intrinsic:>9.2f}")
                        
                    except:
                        continue
                        
            if not call_strikes and not put_strikes:
                print(f"{YELL}No ITM options available (price exactly at strike){RESET}")
                    
            print(f"{CYAN}═════════════════════════════════════════════════{RESET}\n")
            
        except Exception as e:
            print(f"{RED}Error fetching options: {e}{RESET}")
    
    async def run(self):
        try:
            options_counter = 0
            while True:
                await asyncio.sleep(60)  # Check every minute
                options_counter += 1
                
                # Update options chain every 5 minutes
                if SHOW_OPTIONS and options_counter >= 5:
                    options_counter = 0
                    for sym in SYMBOLS:
                        await self.get_0dte_options(sym)
        finally:
            self.agg.flush()
            self.ib.disconnect()
 
    def _on_bar(self, bars, hasNewBar):
        if not hasNewBar: return
        sym = bars.contract.symbol
        b = bars[-1]
        ts = b.time.astimezone(timezone.utc)
        # FIX: Use open_ instead of open (with underscore)
        self.agg.on_bar_5s(sym, ts,
                          float(b.open_), float(b.high), float(b.low),
                          float(b.close), float(b.volume or 0))
 
# ---- Entrypoint ----
def main():
    s = OneMinuteStreamer()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(s.connect())
    loop.run_until_complete(s.run())
 
if __name__ == "__main__":
    main()