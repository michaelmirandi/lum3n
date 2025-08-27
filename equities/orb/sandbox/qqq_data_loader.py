#!/usr/bin/env python3
"""
QQQ Data Loader - Smart data loading system
Tries live IBKR first, falls back to CSV backfill
"""

import pandas as pd
import os
from datetime import datetime
from ib_insync import IB, Stock, util

# Configuration
IB_HOST = '127.0.0.1'
IB_PORT = 7496
CLIENT_ID = 104

class QQQDataLoader:
    def __init__(self):
        self.use_csv_fallback = True
        self.csv_files = {
            '1m': 'qqq_1m_backfill.csv',
            '5m': 'qqq_5m_backfill.csv', 
            '4h': 'qqq_4h_backfill.csv'
        }
    
    def load_from_csv(self, timeframe: str) -> pd.DataFrame:
        """Load data from CSV backfill files with robust date handling"""
        filename = self.csv_files.get(timeframe)
        if filename and os.path.exists(filename):
            print(f"📂 Loading {timeframe} data from {filename}")
            try:
                df = pd.read_csv(filename)
                
                # Handle date conversion robustly
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                
                # Remove any invalid dates
                initial_len = len(df)
                df = df.dropna(subset=['date'])
                df = df[df['date'].dt.year >= 2020]  # Filter out epoch dates
                
                if len(df) < initial_len:
                    print(f"   ⚠️  Filtered out {initial_len - len(df)} invalid dates")
                
                if not df.empty:
                    df = df.sort_values('date').reset_index(drop=True)
                    print(f"   ✅ Loaded {len(df):,} bars ({df['date'].min()} to {df['date'].max()})")
                    return df
                else:
                    print(f"   ❌ No valid data after filtering")
                    return pd.DataFrame()
                    
            except Exception as e:
                print(f"   ❌ Error loading CSV: {e}")
                return pd.DataFrame()
        else:
            print(f"❌ CSV file not found for {timeframe}: {filename}")
            return pd.DataFrame()
    
    def try_live_data(self, symbol='QQQ', timeframe='1m', days=1):
        """Try to fetch live data from IBKR (with timeout protection)"""
        try:
            print(f"🔄 Attempting live {timeframe} data fetch for {days} days...")
            
            ib = IB()
            ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID, timeout=10)  # 10 sec timeout
            
            stock = Stock(symbol, 'ARCA', 'USD')
            ib.qualifyContracts(stock)
            
            # Map timeframe to bar size
            bar_size_map = {
                '1m': '1 min',
                '5m': '5 mins', 
                '4h': '4 hours'
            }
            
            bars = ib.reqHistoricalData(
                stock,
                endDateTime='',
                durationStr=f'{days} D',
                barSizeSetting=bar_size_map.get(timeframe, '1 min'),
                whatToShow='TRADES',
                useRTH=False,
                formatDate=2,  # String format to avoid timezone issues
                timeout=30  # 30 second timeout
            )
            
            if bars:
                df = pd.DataFrame([{
                    'date': pd.to_datetime(str(bar.date)),
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': int(bar.volume),
                    'average': float(bar.average),
                    'barCount': int(bar.barCount)
                } for bar in bars])
                
                print(f"   ✅ Live data: {len(df)} bars")
                ib.disconnect()
                return df
            else:
                print(f"   ⚠️ No live data returned")
                ib.disconnect()
                return pd.DataFrame()
                
        except Exception as e:
            print(f"   ❌ Live data failed: {e}")
            try:
                ib.disconnect()
            except:
                pass
            return pd.DataFrame()
    
    def get_data(self, timeframe: str, prefer_live: bool = False, live_days: int = 1) -> pd.DataFrame:
        """Smart data loading: try live first, fallback to CSV"""
        
        if prefer_live:
            # Try live data first
            live_df = self.try_live_data(timeframe=timeframe, days=live_days)
            if not live_df.empty:
                return live_df
            else:
                print(f"🔄 Live data failed, falling back to CSV...")
        
        # Use CSV backfill data
        csv_df = self.load_from_csv(timeframe)
        if not csv_df.empty:
            return csv_df
        else:
            print(f"❌ Both live and CSV data unavailable for {timeframe}")
            return pd.DataFrame()

def get_qqq_data(prefer_live=False, live_days=14):
    """
    Main function to get QQQ data for all timeframes
    
    Args:
        prefer_live: Try live data first before CSV
        live_days: How many days of live data to fetch
    
    Returns:
        df_1m, df_5m, df_4h: DataFrames for each timeframe
    """
    print("🎯 QQQ SMART DATA LOADER")
    print("=" * 40)
    
    loader = QQQDataLoader()
    
    # Load each timeframe
    df_1m = loader.get_data('1m', prefer_live=prefer_live, live_days=live_days)
    df_5m = loader.get_data('5m', prefer_live=prefer_live, live_days=live_days) 
    df_4h = loader.get_data('4h', prefer_live=prefer_live, live_days=90)  # Need more 4h history
    
    print(f"\n📊 DATA SUMMARY:")
    for name, df in [('1-minute', df_1m), ('5-minute', df_5m), ('4-hour', df_4h)]:
        if not df.empty:
            print(f"   {name}: {len(df):,} bars ({df['date'].min()} to {df['date'].max()})")
        else:
            print(f"   {name}: ❌ No data available")
    
    return df_1m, df_5m, df_4h

# Quick test functions
def quick_csv_load():
    """Quick function to just load CSV data"""
    print("📂 Loading CSV data only...")
    loader = QQQDataLoader()
    return (
        loader.load_from_csv('1m'),
        loader.load_from_csv('5m'), 
        loader.load_from_csv('4h')
    )

def quick_live_test():
    """Quick function to test live connection"""
    print("🔄 Testing live connection...")
    loader = QQQDataLoader()
    test_df = loader.try_live_data(timeframe='5m', days=1)
    if not test_df.empty:
        print("✅ Live connection working!")
        return True
    else:
        print("❌ Live connection failed")
        return False