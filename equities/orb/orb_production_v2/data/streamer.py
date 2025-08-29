#!/usr/bin/env python3
"""
FAST Real-Time Data Streamer with Precise Candle Aggregation
Clean, efficient streaming with correct timing for ORB indicators
"""
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional, Callable
from collections import defaultdict
from ib_insync import IB, Stock
import pytz

class CandleBuffer:
    """Efficient in-memory candle storage with proper timing"""
    
    def __init__(self, symbol: str, max_size: int = 500):
        self.symbol = symbol
        self.max_size = max_size
        
        # DataFrames for each timeframe
        self.df_1m = pd.DataFrame()
        self.df_5m = pd.DataFrame()  
        self.df_4h = pd.DataFrame()
        
        # Current building candles 
        self.current_1m = None
        self.current_5m = None
        self.current_4h = None
        
        self.eastern = pytz.timezone('US/Eastern')
    
    def add_5s_bar(self, timestamp: datetime, open_: float, high: float, 
                   low: float, close: float, volume: float):
        """Process 5s bar and update all timeframes"""
        
        # Use end time of 5s bar for correct minute assignment
        bar_end_time = timestamp + timedelta(seconds=5)
        ts_et = bar_end_time.astimezone(self.eastern)
        minute_start = ts_et.replace(second=0, microsecond=0)
        
        # Update 1m candle
        self._update_1m(minute_start, open_, high, low, close, volume)
        
        # Update 5m candle 
        self._update_5m(minute_start, open_, high, low, close, volume)
        
        # Update 4h candle
        self._update_4h(minute_start, open_, high, low, close, volume)
    
    def _update_1m(self, minute_start: datetime, o: float, h: float, 
                   l: float, c: float, v: float):
        """Update 1m candle - finalize when minute changes"""
        
        if self.current_1m is None or self.current_1m['start'] != minute_start:
            # Finalize previous candle if exists
            if self.current_1m is not None:
                self._finalize_1m()
            
            # Start new 1m candle
            self.current_1m = {
                'start': minute_start,
                'open': o, 'high': h, 'low': l, 'close': c, 'volume': v
            }
        else:
            # Update existing candle
            self.current_1m['high'] = max(self.current_1m['high'], h)
            self.current_1m['low'] = min(self.current_1m['low'], l)
            self.current_1m['close'] = c
            self.current_1m['volume'] += v
    
    def _update_5m(self, minute_start: datetime, o: float, h: float,
                   l: float, c: float, v: float):
        """Update 5m candle - finalize when 5m period changes"""
        
        # Calculate 5m period start (floor to 5m boundary)
        period_start = minute_start.replace(
            minute=(minute_start.minute // 5) * 5
        )
        
        if self.current_5m is None or self.current_5m['start'] != period_start:
            # Finalize previous candle if exists  
            if self.current_5m is not None:
                self._finalize_5m()
            
            # Start new 5m candle
            self.current_5m = {
                'start': period_start,
                'open': o, 'high': h, 'low': l, 'close': c, 'volume': v
            }
        else:
            # Update existing candle
            self.current_5m['high'] = max(self.current_5m['high'], h)
            self.current_5m['low'] = min(self.current_5m['low'], l)
            self.current_5m['close'] = c
            self.current_5m['volume'] += v
    
    def _update_4h(self, minute_start: datetime, o: float, h: float,
                   l: float, c: float, v: float):
        """Update 4h candle - finalize when 4h period changes"""
        
        # Calculate 4h period start
        period_start = minute_start.replace(
            hour=(minute_start.hour // 4) * 4,
            minute=0
        )
        
        if self.current_4h is None or self.current_4h['start'] != period_start:
            # Finalize previous candle if exists
            if self.current_4h is not None:
                self._finalize_4h()
                
            # Start new 4h candle 
            self.current_4h = {
                'start': period_start,
                'open': o, 'high': h, 'low': l, 'close': c, 'volume': v
            }
        else:
            # Update existing candle
            self.current_4h['high'] = max(self.current_4h['high'], h)
            self.current_4h['low'] = min(self.current_4h['low'], l)
            self.current_4h['close'] = c
            self.current_4h['volume'] += v
    
    def _finalize_1m(self):
        """Add completed 1m candle to DataFrame"""
        new_row = pd.DataFrame([{
            'date': self.current_1m['start'],
            'open': self.current_1m['open'],
            'high': self.current_1m['high'],
            'low': self.current_1m['low'], 
            'close': self.current_1m['close'],
            'volume': self.current_1m['volume']
        }])
        
        self.df_1m = pd.concat([self.df_1m, new_row], ignore_index=True)
        
        # Trim to max size
        if len(self.df_1m) > self.max_size:
            self.df_1m = self.df_1m.tail(self.max_size)
            
        # Log candle completion with Eastern time
        display_time_et = datetime.now(self.eastern)
        print(f"1m  | {self.current_1m['start'].strftime('%H:%M')} | {self.symbol} | "
              f"O:${self.current_1m['open']:.2f} H:${self.current_1m['high']:.2f} L:${self.current_1m['low']:.2f} C:${self.current_1m['close']:.2f} V:{self.current_1m['volume']:.0f} | "
              f"DISPLAYED: {display_time_et.strftime('%H:%M:%S.%f')[:-3]} ET")
    
    def _finalize_5m(self):
        """Add completed 5m candle to DataFrame"""
        new_row = pd.DataFrame([{
            'date': self.current_5m['start'],
            'open': self.current_5m['open'],
            'high': self.current_5m['high'],
            'low': self.current_5m['low'],
            'close': self.current_5m['close'], 
            'volume': self.current_5m['volume']
        }])
        
        self.df_5m = pd.concat([self.df_5m, new_row], ignore_index=True)
        
        # Trim to max size  
        if len(self.df_5m) > self.max_size // 5:
            self.df_5m = self.df_5m.tail(self.max_size // 5)
            
        # Log candle completion with Eastern time
        display_time_et = datetime.now(self.eastern)
        print(f"5m  | {self.current_5m['start'].strftime('%H:%M')} | {self.symbol} | "
              f"O:${self.current_5m['open']:.2f} H:${self.current_5m['high']:.2f} L:${self.current_5m['low']:.2f} C:${self.current_5m['close']:.2f} V:{self.current_5m['volume']:.0f} | "
              f"DISPLAYED: {display_time_et.strftime('%H:%M:%S.%f')[:-3]} ET")
    
    def _finalize_4h(self):
        """Add completed 4h candle to DataFrame"""  
        new_row = pd.DataFrame([{
            'date': self.current_4h['start'],
            'open': self.current_4h['open'],
            'high': self.current_4h['high'],
            'low': self.current_4h['low'],
            'close': self.current_4h['close'],
            'volume': self.current_4h['volume']
        }])
        
        self.df_4h = pd.concat([self.df_4h, new_row], ignore_index=True)
        
        # Trim to max size
        if len(self.df_4h) > self.max_size // 48:  # 48 4h candles per day
            self.df_4h = self.df_4h.tail(self.max_size // 48)
            
        # Log candle completion with Eastern time
        display_time_et = datetime.now(self.eastern)
        print(f"4h  | {self.current_4h['start'].strftime('%H:%M')} | {self.symbol} | "
              f"O:${self.current_4h['open']:.2f} H:${self.current_4h['high']:.2f} L:${self.current_4h['low']:.2f} C:${self.current_4h['close']:.2f} V:{self.current_4h['volume']:.0f} | "
              f"DISPLAYED: {display_time_et.strftime('%H:%M:%S.%f')[:-3]} ET")

    def get_dataframes(self) -> tuple:
        """Get current DataFrames for all timeframes"""
        return (
            self.df_1m.copy() if not self.df_1m.empty else pd.DataFrame(),
            self.df_5m.copy() if not self.df_5m.empty else pd.DataFrame(), 
            self.df_4h.copy() if not self.df_4h.empty else pd.DataFrame()
        )
    
    def load_historical(self, df_1m: pd.DataFrame, df_5m: pd.DataFrame, df_4h: pd.DataFrame):
        """Load historical data into buffers"""
        self.df_1m = df_1m.copy()
        self.df_5m = df_5m.copy()
        self.df_4h = df_4h.copy()


class LiveDataStreamer:
    """Fast, clean real-time data streaming"""
    
    def __init__(self, symbols: list, buffer_size: int = 500):
        self.symbols = symbols
        self.buffer_size = buffer_size
        
        # Candle buffers per symbol
        self.buffers = {
            symbol: CandleBuffer(symbol, buffer_size) 
            for symbol in symbols
        }
        
        self.contracts = {}
        self.streaming = False
    
    async def start_streaming(self, ib: IB) -> bool:
        """Start real-time streaming"""
        print("🚀 Starting live data stream...")
        
        # Qualify contracts
        for symbol in self.symbols:
            contract = Stock(symbol, 'ARCA' if symbol == 'QQQ' else 'SMART', 'USD')
            contracts = await ib.qualifyContractsAsync(contract)
            if not contracts:
                print(f"❌ Cannot qualify {symbol}")
                return False
            self.contracts[symbol] = contracts[0]
        
        # Setup bar handler  
        ib.barUpdateEvent += self._on_bar_update
        
        # Request 5s real-time bars
        for symbol in self.symbols:
            ib.reqRealTimeBars(
                self.contracts[symbol],
                5,  # 5 second bars
                'TRADES',
                useRTH=True
            )
            print(f"📡 Streaming {symbol}")
        
        self.streaming = True
        print("✅ Live streaming active")
        return True
    
    def _on_bar_update(self, bars, hasNewBar):
        """Handle new bar from IB"""
        if not hasNewBar or not bars:
            return
            
        symbol = bars.contract.symbol
        bar = bars[-1]  # Latest bar
        
        # Log 5s pulse receipt time in Eastern
        eastern = pytz.timezone('US/Eastern')
        pulse_received_et = datetime.now(eastern)
        print(f"5s PULSE | {pulse_received_et.strftime('%H:%M:%S.%f')[:-3]} ET | {symbol} | Bar time: {bar.time}")
        
        # Feed to candle buffer
        if symbol in self.buffers:
            self.buffers[symbol].add_5s_bar(
                bar.time,
                float(bar.open_),
                float(bar.high),
                float(bar.low), 
                float(bar.close),
                float(bar.volume or 0)
            )
    
    def get_dataframes(self, symbol: str) -> tuple:
        """Get current DataFrames for symbol"""
        if symbol in self.buffers:
            return self.buffers[symbol].get_dataframes()
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    def load_historical_data(self, historical_data: Dict):
        """Load historical data into buffers before streaming"""
        for symbol, data in historical_data.items():
            if symbol in self.buffers:
                self.buffers[symbol].load_historical(
                    data.get('1m', pd.DataFrame()),
                    data.get('5m', pd.DataFrame()), 
                    data.get('4h', pd.DataFrame())
                )
                print(f"📊 Loaded {symbol}: "
                      f"{len(data.get('1m', []))} 1m, "
                      f"{len(data.get('5m', []))} 5m, " 
                      f"{len(data.get('4h', []))} 4h bars")
    
    async def stop_streaming(self, ib: IB):
        """Stop streaming and cleanup"""
        print("🛑 Stopping data stream...")
        
        for symbol, contract in self.contracts.items():
            try:
                ib.cancelRealTimeBars(contract)
                print(f"✅ Stopped {symbol}")
            except Exception as e:
                print(f"⚠️ Error stopping {symbol}: {e}")
        
        self.streaming = False
        print("✅ Stream stopped")