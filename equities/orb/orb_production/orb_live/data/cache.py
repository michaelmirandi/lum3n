#!/usr/bin/env python3
"""
Historical Data Cache Manager
Handles pre-market data loading for technical indicators
"""
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import pickle
from typing import Dict, Optional, Tuple
from ib_insync import IB, Stock, BarData

class DataCache:
    """
    Manages historical data caching for technical indicator calculations
    Fetches and stores previous day's data to ensure indicators are ready at market open
    """
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ib = None
        
    def get_cache_filename(self, symbol: str, date: datetime) -> Path:
        """Generate cache filename for symbol and date"""
        date_str = date.strftime("%Y%m%d")
        return self.cache_dir / f"{symbol}_{date_str}.pkl"
    
    async def fetch_historical_data(self, ib: IB, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data using proven chunked approach from qqq_data_backfill.py
        """
        print(f"📥 Fetching historical data for {symbol} using proven chunked method...")
        
        # Use proper contract (matches working backfill)
        if symbol == 'QQQ':
            contract = Stock(symbol, 'ARCA', 'USD')
        else:
            contract = Stock(symbol, 'SMART', 'USD')
        
        # Qualify contract
        contracts = await ib.qualifyContractsAsync(contract)
        if not contracts:
            raise ValueError(f"Could not qualify contract for {symbol}")
        contract = contracts[0]
        print(f"  ✅ {symbol} contract qualified")
        
        data = {}
        
        # Fetch parameters (matching working backfill exactly)
        backfill_params = [
            ('1 min', 10, 2, '1m'),    # 10 days in 2-day chunks
            ('5 mins', 15, 3, '5m'),   # 15 days in 3-day chunks
            ('4 hours', 60, 10, '4h')  # 60 days in 10-day chunks
        ]
        
        for bar_size, total_days, chunk_days, key in backfill_params:
            print(f"  📊 Fetching {bar_size} data ({total_days} days in {chunk_days}-day chunks)")
            
            all_data = []
            current_date = datetime.now()
            num_chunks = (total_days + chunk_days - 1) // chunk_days
            
            for chunk_num in range(num_chunks):
                chunk_end = current_date - timedelta(days=chunk_num * chunk_days)
                end_date_str = chunk_end.strftime('%Y%m%d %H:%M:%S')
                
                try:
                    print(f"    📥 Chunk {chunk_num + 1}/{num_chunks} ending {chunk_end.strftime('%Y-%m-%d')}")
                    
                    bars = await ib.reqHistoricalDataAsync(
                        contract,
                        endDateTime=end_date_str,
                        durationStr=f'{chunk_days} D',
                        barSizeSetting=bar_size,
                        whatToShow='TRADES',
                        useRTH=True,  # MARKET HOURS ONLY - key difference!
                        formatDate=1
                    )
                    
                    if bars:
                        chunk_df = self._bars_to_dataframe_validated(bars)
                        if not chunk_df.empty:
                            all_data.append(chunk_df)
                            print(f"    ✅ Got {len(chunk_df)} valid bars")
                    
                    # Rate limiting (crucial for IBKR)
                    if chunk_num < num_chunks - 1:
                        print(f"    ⏳ Waiting 1 seconds...")
                        await asyncio.sleep(1)
                        
                except Exception as e:
                    print(f"    ❌ Error fetching chunk: {e}")
                    continue
            
            # Combine chunks
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
                data[key] = combined_df
                print(f"  ✅ {key} combined: {len(combined_df)} bars")
            else:
                print(f"  ❌ No data collected for {key}")
                data[key] = pd.DataFrame()
        
        return data
    
    def _bars_to_dataframe_validated(self, bars: list) -> pd.DataFrame:
        """Convert IBKR bars to DataFrame with validation - matches working backfill exactly"""
        data_list = []
        for bar in bars:
            # Handle datetime properly (same as working backfill)
            bar_date = bar.date
            
            if isinstance(bar_date, str):
                try:
                    if ' ' in bar_date:
                        parsed_date = pd.to_datetime(bar_date, format='%Y%m%d %H:%M:%S')
                    else:
                        parsed_date = pd.to_datetime(bar_date, format='%Y%m%d')
                except:
                    parsed_date = pd.to_datetime(bar_date)
            else:
                parsed_date = pd.Timestamp(bar_date)
            
            # Timezone handling (same as working backfill)
            if parsed_date.tz is not None:
                if parsed_date.tz != 'US/Eastern':
                    parsed_date = parsed_date.tz_convert('US/Eastern')
                parsed_date = parsed_date.tz_localize(None)
            
            # Skip invalid dates and weekends
            if parsed_date.year < 2020 or parsed_date.weekday() > 4:
                continue
            
            # Market hours validation
            market_start = pd.Timestamp('09:30:00').time()
            market_end = pd.Timestamp('16:00:00').time()
            if parsed_date.time() < market_start or parsed_date.time() > market_end:
                continue
            
            data_list.append({
                'date': parsed_date,
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': int(bar.volume)
            })
        
        if data_list:
            df = pd.DataFrame(data_list)
            return df
        else:
            return pd.DataFrame()
    
    def _bars_to_dataframe(self, bars: list) -> pd.DataFrame:
        """Legacy method - keeping for compatibility"""
        return self._bars_to_dataframe_validated(bars)
    
    def save_cache(self, symbol: str, date: datetime, data: Dict[str, pd.DataFrame]):
        """Save historical data to cache file"""
        filename = self.get_cache_filename(symbol, date)
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Cached data saved to {filename}")
    
    def load_cache(self, symbol: str, date: datetime) -> Optional[Dict[str, pd.DataFrame]]:
        """Load historical data from cache if available"""
        filename = self.get_cache_filename(symbol, date)
        if filename.exists():
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            print(f"📂 Loaded cached data from {filename}")
            return data
        return None
    
    def merge_with_live_data(self, cached_data: Dict[str, pd.DataFrame],
                           live_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Merge cached historical data with live streaming data
        Ensures continuity for technical indicators
        """
        merged = {}
        
        for timeframe in ['1m', '5m', '4h']:
            if timeframe in cached_data:
                cached_df = cached_data[timeframe].copy()
                
                if timeframe in live_data and not live_data[timeframe].empty:
                    live_df = live_data[timeframe].copy()
                    
                    # Remove any overlapping data (keep live version)
                    if not cached_df.empty and not live_df.empty:
                        last_cached_time = cached_df['date'].max()
                        first_live_time = live_df['date'].min()
                        
                        if first_live_time <= last_cached_time:
                            # Trim cached data to avoid overlap
                            cached_df = cached_df[cached_df['date'] < first_live_time]
                    
                    # Concatenate cached and live data
                    merged[timeframe] = pd.concat([cached_df, live_df], ignore_index=True)
                else:
                    merged[timeframe] = cached_df
            elif timeframe in live_data:
                merged[timeframe] = live_data[timeframe]
            else:
                merged[timeframe] = pd.DataFrame()
        
        return merged
    
    async def initialize_for_trading(self, ib: IB, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Initialize data for trading day
        1. Check for today's cache
        2. If not available, fetch historical data
        3. Return data ready for indicator calculations
        """
        today = datetime.now().date()
        
        # Try to load cached data
        cached_data = self.load_cache(symbol, today)
        
        if cached_data is None:
            # Fetch fresh historical data
            cached_data = await self.fetch_historical_data(ib, symbol)
            # Save to cache for later use
            self.save_cache(symbol, today, cached_data)
        
        return cached_data
    
    def get_lookback_requirements(self) -> Dict[str, int]:
        """
        Define minimum bars needed for each indicator
        Used to validate we have enough historical data
        """
        return {
            '1m': {
                'rsi': 14,      # RSI period
                'ema': 9,       # EMA period
                'macd': 26,     # MACD slow period
                'volume_ma': 20 # Volume MA period
            },
            '5m': {
                'atr': 14,      # ATR period
                'ema_21': 21    # 21-period EMA
            },
            '4h': {
                'rsi': 14,      # 4H RSI
                'macd': 26,     # 4H MACD
                'ema_20': 20    # 4H EMA
            }
        }
    
    def validate_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """
        Validate that we have enough historical data for indicators
        """
        requirements = self.get_lookback_requirements()
        
        for timeframe, indicators in requirements.items():
            if timeframe not in data or data[timeframe].empty:
                print(f"❌ Missing {timeframe} data")
                return False
            
            df = data[timeframe]
            bars_available = len(df)
            max_required = max(indicators.values())
            
            if bars_available < max_required:
                print(f"❌ Insufficient {timeframe} data: {bars_available} bars, need {max_required}")
                return False
            
            print(f"✅ {timeframe}: {bars_available} bars available (need {max_required})")
        
        return True