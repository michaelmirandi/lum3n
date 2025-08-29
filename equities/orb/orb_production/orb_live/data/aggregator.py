#!/usr/bin/env python3
"""
Candle Aggregator - Converts 5s bars to 1m, 5m, 15m, 4h candles
Direct port from ibkr.py with enhanced buffering
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional

# ANSI colors for console output
RED   = "\033[31m"
GREEN = "\033[32m"
YELL  = "\033[33m"
BLUE  = "\033[34m"
MAG   = "\033[35m"
CYAN  = "\033[36m"
RESET = "\033[0m"

class CandleAggregator:
    """
    Aggregates 5s bars into multiple timeframes and maintains DataFrames
    Enhanced with buffering for technical indicator calculations
    """
    def __init__(self, buffer_minutes: int = 390):  # 6.5 hours of 1m data
        # Rolling state for current candles per symbol
        self.state_1m = {}
        self.state_5m = {}
        self.state_15m = {}
        self.state_4h = {}
        
        # Track last price for each symbol
        self.last_price = {}
        
        # DataFrames for each timeframe (keep last N minutes of data)
        self.buffer_minutes = buffer_minutes
        self.df_1m = {}  # symbol -> DataFrame
        self.df_5m = {}
        self.df_15m = {}
        self.df_4h = {}
        
    def on_bar_5s(self, sym: str, ts_utc: datetime, o: float, h: float, 
                  l: float, c: float, v: float):
        """Process incoming 5-second bar and aggregate"""
        # Track last price
        self.last_price[sym] = c
        
        # NORMALIZE TO EASTERN TIME IMMEDIATELY
        import pytz
        eastern = pytz.timezone('US/Eastern')
        # Convert UTC timestamp to Eastern
        ts_eastern = ts_utc.replace(tzinfo=pytz.UTC).astimezone(eastern).replace(tzinfo=None)
        
        # LOG EVERY 5s BAR in Eastern
        # ts_str = ts_eastern.strftime("%Y-%m-%d %H:%M:%S")
        # col = GREEN if c > o else RED if c < o else YELL
        # print(f"5s  | {ts_str} | {sym:<5} "
        #       f"O:{o:8.2f} H:{h:8.2f} L:{l:8.2f} "
        #       f"C:{col}{c:8.2f}{RESET} V:{v:10.0f}")
        
        # Round down to minute for 1m aggregation - use Eastern time
        min_key = ts_eastern.replace(second=0, microsecond=0)
        
        # 1 Minute candle aggregation
        st = self.state_1m.get(sym)
        if st is None or st["minute"] != min_key:
            if st is not None:
                # Finalize completed 1m candle
                self._finalize_and_store_1m(sym, st)
                
                # Check if this 1m completion triggers 5m/4h completion
                self._check_timeframe_completion_on_1m_close(sym, st)
                
                # Always aggregate into ongoing 5m candle
                self._aggregate_5m(sym, st)
            self.state_1m[sym] = {
                "minute": min_key, "o": o, "h": h, 
                "l": l, "c": c, "v": v
            }
        else:
            st["c"] = c
            st["h"] = max(st["h"], h)
            st["l"] = min(st["l"], l)
            st["v"] += v
    
    def _check_timeframe_completion_on_1m_close(self, sym: str, completed_1m_candle: dict):
        """Check if the completed 1m candle triggers 5m or 4h candle completion"""
        completed_minute = completed_1m_candle["minute"]
        
        # Check if this 1m completion ends a 5m period
        # 5m periods: 9:30-9:34, 9:35-9:39, etc.
        # So completion at 9:34 should finalize the 9:30-9:34 period
        next_minute = completed_minute + timedelta(minutes=1)
        if next_minute.minute % 5 == 0:  # Next minute starts new 5m period
            st5 = self.state_5m.get(sym)
            if st5:  # Only finalize if we have a 5m candle to finalize
                self._finalize_and_store_5m(sym, st5)
                self._aggregate_15m(sym, st5)
        
        # Check if this 1m completion ends a 4h period  
        # 4h periods: 4:00-7:59, 8:00-11:59, etc.
        # So completion at 7:59 should finalize the 4:00-7:59 period
        next_minute = completed_minute + timedelta(minutes=1)
        if next_minute.hour % 4 == 0 and next_minute.minute == 0:  # Next minute starts new 4h period
            st4h = self.state_4h.get(sym)
            if st4h:  # Only finalize if we have a 4h candle to finalize
                self._finalize_and_store_4h(sym, st4h)
    
    def _aggregate_5m(self, sym: str, one_min_candle: dict):
        """Aggregate 1m candle into 5m candle"""
        min_key = one_min_candle["minute"]
        floored = min_key.replace(minute=(min_key.minute // 5) * 5, second=0, microsecond=0)
        
        st5 = self.state_5m.get(sym)
        if st5 is None or st5["minute"] != floored:
            self.state_5m[sym] = {
                "minute": floored,
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
    
    def _aggregate_15m(self, sym: str, five_min_candle: dict):
        """Aggregate 5m candle into 15m candle"""
        min_key = five_min_candle["minute"]
        floored = min_key.replace(minute=(min_key.minute // 15) * 15, second=0, microsecond=0)
        
        st15 = self.state_15m.get(sym)
        if st15 is None or st15["minute"] != floored:
            self.state_15m[sym] = {
                "minute": floored,
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
    
    def _aggregate_4h(self, sym: str, fifteen_min_candle: dict):
        """Aggregate 15m candle into 4h candle"""
        min_key = fifteen_min_candle["minute"]
        floored = min_key.replace(hour=(min_key.hour // 4) * 4, minute=0, second=0, microsecond=0)
        
        st4h = self.state_4h.get(sym)
        if st4h is None or st4h["hour"] != floored:
            self.state_4h[sym] = {
                "hour": floored,
                "o": fifteen_min_candle["o"],
                "h": fifteen_min_candle["h"],
                "l": fifteen_min_candle["l"],
                "c": fifteen_min_candle["c"],
                "v": fifteen_min_candle["v"],
            }
        else:
            st4h["c"] = fifteen_min_candle["c"]
            st4h["h"] = max(st4h["h"], fifteen_min_candle["h"])
            st4h["l"] = min(st4h["l"], fifteen_min_candle["l"])
            st4h["v"] += fifteen_min_candle["v"]
    
    def _finalize_and_store_1m(self, sym: str, candle: dict):
        """Log and store completed 1m candle in DataFrame"""
        self._log(sym, candle, "1m")
        self._append_to_dataframe(sym, candle, self.df_1m, "1m")
    
    def _finalize_and_store_5m(self, sym: str, candle: dict):
        """Log and store completed 5m candle in DataFrame"""
        self._log(sym, candle, "5m")
        self._append_to_dataframe(sym, candle, self.df_5m, "5m")
    
    def _finalize_and_store_4h(self, sym: str, candle: dict):
        """Log and store completed 4h candle in DataFrame"""
        ts = candle["hour"].strftime("%Y-%m-%d %H:%M")
        o,h,l,c,v = candle["o"], candle["h"], candle["l"], candle["c"], candle["v"]
        col = GREEN if c > o else RED if c < o else YELL
        print(f"4h  | {ts} | {sym:<5} "
              f"O:{o:8.2f} H:{h:8.2f} L:{l:8.2f} "
              f"C:{col}{c:8.2f}{RESET} V:{v:10.0f}")
        
        # Add to 4h DataFrame
        if sym not in self.df_4h:
            self.df_4h[sym] = pd.DataFrame()
        
        new_row = pd.DataFrame([{
            'date': candle["hour"],
            'open': candle["o"],
            'high': candle["h"],
            'low': candle["l"],
            'close': candle["c"],
            'volume': candle["v"]
        }])
        
        self.df_4h[sym] = pd.concat([self.df_4h[sym], new_row], ignore_index=True)
    
    def _log(self, sym: str, st: dict, tf: str):
        """Log completed candle with color coding"""
        ts = st["minute"].strftime("%Y-%m-%d %H:%M")
        o,h,l,c,v = st["o"], st["h"], st["l"], st["c"], st["v"]
        col = GREEN if c > o else RED if c < o else YELL
        print(f"{tf:<3} | {ts} | {sym:<5} "
              f"O:{o:8.2f} H:{h:8.2f} L:{l:8.2f} "
              f"C:{col}{c:8.2f}{RESET} V:{v:10.0f}")
    
    def _append_to_dataframe(self, sym: str, candle: dict, 
                           df_dict: dict, timeframe: str):
        """Append candle to appropriate DataFrame"""
        if sym not in df_dict:
            df_dict[sym] = pd.DataFrame()
        
        new_row = pd.DataFrame([{
            'date': candle["minute"],
            'open': candle["o"],
            'high': candle["h"],
            'low': candle["l"],
            'close': candle["c"],
            'volume': candle["v"]
        }])
        
        df_dict[sym] = pd.concat([df_dict[sym], new_row], ignore_index=True)
        
        # Trim to buffer size (keep last N minutes)
        if timeframe == "1m" and len(df_dict[sym]) > self.buffer_minutes:
            df_dict[sym] = df_dict[sym].iloc[-self.buffer_minutes:]
        elif timeframe == "5m" and len(df_dict[sym]) > self.buffer_minutes // 5:
            df_dict[sym] = df_dict[sym].iloc[-(self.buffer_minutes // 5):]
        elif timeframe == "15m" and len(df_dict[sym]) > self.buffer_minutes // 15:
            df_dict[sym] = df_dict[sym].iloc[-(self.buffer_minutes // 15):]
    
    def get_dataframes(self, sym: str) -> tuple:
        """Get DataFrames for a symbol"""
        df_1m = self.df_1m.get(sym, pd.DataFrame())
        df_5m = self.df_5m.get(sym, pd.DataFrame())
        df_4h = self.df_4h.get(sym, pd.DataFrame())
        return df_1m, df_5m, df_4h
    
    def flush(self):
        """Flush all remaining candles at end of day"""
        for sym in list(self.state_1m.keys()):
            if sym in self.state_1m:
                self._finalize_and_store_1m(sym, self.state_1m[sym])
                self._aggregate_5m(sym, self.state_1m[sym])
        
        for sym in list(self.state_5m.keys()):
            if sym in self.state_5m:
                self._finalize_and_store_5m(sym, self.state_5m[sym])
                self._aggregate_15m(sym, self.state_5m[sym])
        
        self.state_1m.clear()
        self.state_5m.clear()
        self.state_15m.clear()