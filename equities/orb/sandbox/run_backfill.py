#!/usr/bin/env python3
"""
Quick Backfill Execution Script
Run this to populate your CSV files with historical data
"""

import asyncio
import sys
from datetime import datetime
from qqq_data_backfill import QQQDataBackfill
from ib_insync import util

async def quick_backfill():
    """Quick backfill with conservative parameters to avoid timeouts"""
    
    print("🚀 QUICK QQQ DATA BACKFILL")
    print("⏱️ Using conservative parameters to avoid timeouts")
    print("=" * 60)
    
    backfill = QQQDataBackfill()
    
    try:
        # Connect
        await backfill.connect()
        
        # Conservative backfill parameters (smaller chunks, more delays)
        backfill_jobs = [
            # (bar_size, total_days, chunk_days, filename)
            ('1 min', 365, 5, 'qqq_1m_backfill.csv'),      # 14 days in 1-day chunks
            ('5 mins', 365, 5, 'qqq_5m_backfill.csv'),     # 30 days in 3-day chunks
            ('4 hours', 380, 15, 'qqq_4h_backfill.csv'),   # 90 days in 15-day chunks
        ]
        
        for bar_size, total_days, chunk_days, filename in backfill_jobs:
            print(f"\n⏳ Starting {bar_size} backfill...")
            df = await backfill.backfill_timeframe(bar_size, total_days, chunk_days, filename)
            
            if not df.empty:
                print(f"✅ {bar_size} complete: {len(df):,} bars saved to {filename}")
            else:
                print(f"❌ {bar_size} failed")
            
            # Extra delay between timeframes
            print("⏳ Waiting 5 seconds before next timeframe...")
            await asyncio.sleep(5)
    
    except Exception as e:
        print(f"❌ Backfill error: {e}")
    
    finally:
        backfill.disconnect()
    
    print(f"\n🎯 BACKFILL COMPLETE!")
    print(f"💡 Now you can use quick_csv_load() in your notebooks")

def main():
    """Main execution"""
    # For notebooks compatibility
    util.startLoop()
    
    # Run the backfill
    asyncio.run(quick_backfill())

if __name__ == "__main__":
    main()