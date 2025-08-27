#!/usr/bin/env python3
"""
Simple QQQ Backfill - Handles IBKR timezone issues robustly
"""

from ib_insync import IB, Stock, util
import pandas as pd
from datetime import datetime, timedelta
import time

# Configuration
IB_HOST = '127.0.0.1'
IB_PORT = 7496
CLIENT_ID = 105

def clean_bar_data(bars):
    """Clean and validate bar data from IBKR"""
    clean_data = []
    
    for bar in bars:
        try:
            # Get the date - handle different IBKR formats
            bar_date = bar.date
            
            # Convert to pandas timestamp and handle timezones
            if isinstance(bar_date, str):
                # String format from IBKR
                if len(bar_date) == 8:  # Daily format: "20241201"
                    dt = pd.to_datetime(bar_date, format='%Y%m%d')
                else:  # Intraday format: "20241201  09:30:00"
                    dt = pd.to_datetime(bar_date.strip(), format='%Y%m%d  %H:%M:%S')
            else:
                # Datetime object
                dt = pd.Timestamp(bar_date)
            
            # Remove timezone if present
            if hasattr(dt, 'tz') and dt.tz is not None:
                dt = dt.tz_localize(None)
            
            # Validate date (skip if before 2020 - indicates bad data)
            if dt.year < 2020:
                print(f"    ⚠️ Skipping invalid date: {dt}")
                continue
            
            clean_data.append({
                'date': dt,
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': int(bar.volume)
            })
            
        except Exception as e:
            print(f"    ⚠️ Skipping bad bar: {e}")
            continue
    
    return clean_data

def fetch_qqq_data(bar_size: str, days: int) -> pd.DataFrame:
    """Fetch QQQ data for a specific timeframe"""
    ib = IB()
    
    try:
        # Connect with timeout
        print(f"🔌 Connecting to IBKR...")
        ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID, timeout=15)
        
        # Set up QQQ contract
        qqq = Stock('QQQ', 'ARCA', 'USD')
        ib.qualifyContracts(qqq)
        print(f"✅ Connected and qualified QQQ")
        
        # Fetch data
        print(f"📥 Fetching {bar_size} data for {days} days...")
        bars = ib.reqHistoricalData(
            qqq,
            endDateTime='',  # Now
            durationStr=f'{days} D',
            barSizeSetting=bar_size,
            whatToShow='TRADES',
            useRTH=False,
            formatDate=2,  # String format
            timeout=60
        )
        
        if bars:
            # Clean the data
            clean_data = clean_bar_data(bars)
            
            if clean_data:
                df = pd.DataFrame(clean_data)
                df = df.sort_values('date').reset_index(drop=True)
                print(f"    ✅ Got {len(df)} clean bars")
                print(f"    📅 Range: {df['date'].min()} to {df['date'].max()}")
                return df
            else:
                print(f"    ❌ No valid data after cleaning")
                return pd.DataFrame()
        else:
            print(f"    ❌ No bars returned")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Error fetching {bar_size} data: {e}")
        return pd.DataFrame()
    
    finally:
        try:
            ib.disconnect()
        except:
            pass

def backfill_all_data():
    """Backfill all required timeframes"""
    print("🚀 SIMPLE QQQ BACKFILL")
    print("=" * 40)
    
    # Data requirements
    timeframes = [
        ('1 min', 30, 'qqq_1m_backfill.csv'),
        ('5 mins', 60, 'qqq_5m_backfill.csv'),
        ('4 hours', 120, 'qqq_4h_backfill.csv')
    ]
    
    results = {}
    
    for bar_size, days, filename in timeframes:
        print(f"\n📊 Processing {bar_size}...")
        
        # Fetch data
        df = fetch_qqq_data(bar_size, days)
        
        if not df.empty:
            # Save to CSV
            df.to_csv(filename, index=False)
            results[bar_size] = len(df)
            print(f"    💾 Saved {len(df)} bars to {filename}")
        else:
            results[bar_size] = 0
            print(f"    ❌ Failed to get {bar_size} data")
        
        # Wait between requests
        if bar_size != '4 hours':  # Don't wait after last one
            print(f"    ⏳ Waiting 3 seconds...")
            time.sleep(3)
    
    # Summary
    print(f"\n🎯 BACKFILL SUMMARY:")
    for bar_size, count in results.items():
        status = "✅" if count > 0 else "❌"
        print(f"   {bar_size}: {count} bars {status}")
    
    total_success = sum(1 for count in results.values() if count > 0)
    print(f"\n📈 Success: {total_success}/3 timeframes")
    
    if total_success == 3:
        print("🎉 All data successfully backfilled!")
    else:
        print("⚠️  Some timeframes failed - check IBKR connection")

if __name__ == "__main__":
    backfill_all_data()