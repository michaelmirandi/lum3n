#!/usr/bin/env python3
"""
QQQ Data Backfill System
Fetches 1m, 5m, and 4h data in manageable chunks to avoid IBKR timeouts
Saves to CSV files for reliable loading
"""

import asyncio
from datetime import datetime, timedelta
from ib_insync import IB, Stock, util
import pandas as pd
import os

# Configuration
IB_HOST = '127.0.0.1'
IB_PORT = 7496       # 7497 = paper, 7496 = live
CLIENT_ID = 103      # Different from your main client

class QQQDataBackfill:
    def __init__(self):
        self.ib = None
        self.qqq = None
        
    async def connect(self):
        """Connect to IBKR"""
        self.ib = IB()
        await self.ib.connectAsync(IB_HOST, IB_PORT, clientId=CLIENT_ID)
        print("✅ Connected to IBKR")
        
        # Qualify QQQ contract
        [self.qqq] = await self.ib.qualifyContractsAsync(Stock('QQQ', 'ARCA', 'USD'))
        print("✅ QQQ contract qualified")
    
    def disconnect(self):
        """Disconnect from IBKR"""
        if self.ib:
            self.ib.disconnect()
            print("✅ Disconnected from IBKR")
    
    async def fetch_data_chunk(self, end_date: str, duration: str, bar_size: str):
        """Fetch a single chunk of data"""
        try:
            print(f"  📥 Fetching {bar_size} data ending {end_date} for {duration}")
            
            bars = await self.ib.reqHistoricalDataAsync(
                self.qqq,
                endDateTime=end_date,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,   # MARKET HOURS ONLY (9:30 AM - 4:00 PM EST)
                formatDate=1   # Use datetime objects for proper timezone handling
            )
            
            # Convert to DataFrame with proper datetime handling
            if bars:
                data_list = []
                for bar in bars:
                    # Handle datetime object from IBKR (should be EST/EDT with useRTH=True)
                    bar_date = bar.date
                    
                    if isinstance(bar_date, str):
                        # Parse string format if needed
                        try:
                            if ' ' in bar_date:
                                parsed_date = pd.to_datetime(bar_date, format='%Y%m%d %H:%M:%S')
                            else:
                                parsed_date = pd.to_datetime(bar_date, format='%Y%m%d')
                        except:
                            parsed_date = pd.to_datetime(bar_date)
                    else:
                        # Handle datetime object - IBKR returns in EST/EDT for US markets
                        parsed_date = pd.Timestamp(bar_date)
                    
                    # Convert timezone properly if present
                    if parsed_date.tz is not None:
                        # If timezone-aware, convert to Eastern time first, then remove TZ
                        if parsed_date.tz != 'US/Eastern':
                            # Convert UTC or other timezone to Eastern
                            parsed_date = parsed_date.tz_convert('US/Eastern')
                        # Remove timezone info to make it timezone-naive Eastern time
                        parsed_date = parsed_date.tz_localize(None)
                    
                    # Skip invalid dates and weekend data
                    if parsed_date.year < 2020 or parsed_date.weekday() > 4:  # Skip weekends
                        continue
                    
                    # Additional validation: ensure time is within market hours
                    if parsed_date.time() < pd.Timestamp('09:30:00').time() or parsed_date.time() > pd.Timestamp('16:00:00').time():
                        print(f"    ⚠️ Skipping non-market hours data: {parsed_date}")
                        continue
                    
                    data_list.append({
                        'date': parsed_date,
                        'open': float(bar.open),
                        'high': float(bar.high),
                        'low': float(bar.low),
                        'close': float(bar.close),
                        'volume': int(bar.volume),
                        'average': float(bar.average),
                        'barCount': int(bar.barCount)
                    })
                
                if data_list:
                    df = pd.DataFrame(data_list)
                    print(f"    ✅ Got {len(df)} valid bars ({df['date'].min()} to {df['date'].max()})")
                    return df
                else:
                    print(f"    ⚠️ No valid data after filtering")
                    return pd.DataFrame()
            else:
                print(f"    ⚠️ No data returned")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"    ❌ Error fetching data: {e}")
            return pd.DataFrame()
    
    async def backfill_timeframe(self, bar_size: str, total_days: int, chunk_days: int, filename: str):
        """Backfill data for a specific timeframe in chunks"""
        print(f"\n🔄 BACKFILLING {bar_size.upper()} DATA")
        print(f"📅 Total period: {total_days} days in {chunk_days}-day chunks")
        print("=" * 60)
        
        all_data = []
        current_date = datetime.now()
        
        # Calculate number of chunks needed
        num_chunks = (total_days + chunk_days - 1) // chunk_days  # Round up
        
        for chunk_num in range(num_chunks):
            # Calculate end date for this chunk
            chunk_end = current_date - timedelta(days=chunk_num * chunk_days)
            end_date_str = chunk_end.strftime('%Y%m%d %H:%M:%S')
            print(end_date_str, 'end_date_str')
            
            # Fetch this chunk
            df_chunk = await self.fetch_data_chunk(
                end_date=end_date_str,
                duration=f'{chunk_days} D',
                bar_size=bar_size
            )
            
            if not df_chunk.empty:
                all_data.append(df_chunk)
            
            # Rate limiting - wait between requests
            if chunk_num < num_chunks - 1:  # Don't wait after last chunk
                print(f"  ⏳ Waiting 1 seconds before next chunk...")
                await asyncio.sleep(1)
        
        # Combine all chunks
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Remove duplicates and sort
            combined_df = combined_df.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
            
            # Save to CSV
            combined_df.to_csv(filename, index=False)
            
            print(f"\n📊 {bar_size.upper()} DATA SUMMARY:")
            print(f"   Total bars: {len(combined_df):,}")
            print(f"   Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
            print(f"   Days covered: {(combined_df['date'].max() - combined_df['date'].min()).days}")
            print(f"   💾 Saved to: {filename}")
            
            return combined_df
        else:
            print(f"❌ No data collected for {bar_size}")
            return pd.DataFrame()

async def main():
    """Main backfill process"""
    print("🚀 QQQ DATA BACKFILL SYSTEM")
    print("=" * 50)
    
    # Initialize backfill system
    backfill = QQQDataBackfill()
    
    try:
        # Connect to IBKR
        await backfill.connect()
        
        # Backfill parameters - adjust based on IBKR limits
        backfill_params = [
            # (bar_size, total_days, chunk_days, filename)
            ('1 min', 20, 2, 'qqq_1m_backfill.csv'),      # 30 days in 2-day chunks
            ('5 mins', 20, 5, 'qqq_5m_backfill.csv'),     # 60 days in 5-day chunks  
            ('4 hours', 20, 10, 'qqq_4h_backfill.csv'),  # 180 days in 30-day chunks
        ]
        
        # Process each timeframe
        results = {}
        for bar_size, total_days, chunk_days, filename in backfill_params:
            df = await backfill.backfill_timeframe(bar_size, total_days, chunk_days, filename)
            results[bar_size] = df
        
        # Summary report
        print(f"\n🎯 BACKFILL COMPLETE!")
        print("=" * 50)
        for bar_size, df in results.items():
            if not df.empty:
                print(f"✅ {bar_size}: {len(df):,} bars saved")
            else:
                print(f"❌ {bar_size}: Failed to collect data")
        
    except Exception as e:
        print(f"❌ Error in main process: {e}")
    
    finally:
        # Always disconnect
        backfill.disconnect()

def run_sync_backfill():
    """Synchronous wrapper for Jupyter compatibility"""
    print("🔄 Starting synchronous backfill...")
    
    # For Jupyter notebooks, use util.startLoop()
    util.startLoop()
    
    # Run the async main function
    asyncio.run(main())

def load_backfilled_data():
    """Load the backfilled data from CSV files"""
    print("📂 LOADING BACKFILLED DATA")
    print("=" * 30)
    
    data = {}
    files = [
        ('1m', 'qqq_1m_backfill.csv'),
        ('5m', 'qqq_5m_backfill.csv'), 
        ('4h', 'qqq_4h_backfill.csv')
    ]
    
    for timeframe, filename in files:
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df['date'] = pd.to_datetime(df['date'])
            data[timeframe] = df
            print(f"✅ {timeframe}: {len(df):,} bars ({df['date'].min()} to {df['date'].max()})")
        else:
            print(f"❌ {timeframe}: File {filename} not found")
            data[timeframe] = pd.DataFrame()
    
    return data['1m'], data['5m'], data['4h']

if __name__ == "__main__":
    # Run the backfill
    run_sync_backfill()
    
    # Test loading the data
    print("\n" + "="*50)
    df_1m, df_5m, df_4h = load_backfilled_data()