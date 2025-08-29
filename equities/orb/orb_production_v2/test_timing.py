#!/usr/bin/env python3
"""
Test Candle Timing Verification
Validates that candles have correct period start timestamps
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pytz

# Add data module to path
sys.path.insert(0, str(Path(__file__).parent))

from data.streamer import CandleBuffer

def test_candle_timing():
    """Test that candle timestamps represent period START"""
    print("🧪 Testing Candle Timing Logic")
    print("=" * 40)
    
    buffer = CandleBuffer("TEST", max_size=100)
    eastern = pytz.timezone('US/Eastern')
    
    # Test scenario: 5s bars coming in during 14:39 minute
    base_time = datetime(2024, 1, 15, 14, 39, 0)  # 14:39:00
    base_time = eastern.localize(base_time)
    
    print("📊 Simulating 5s bars during 14:39 minute...")
    
    # Simulate bars every 5 seconds during the minute
    for seconds in [5, 10, 15, 20, 25, 30]:
        bar_time = base_time + timedelta(seconds=seconds)
        print(f"  📈 5s bar at {bar_time.strftime('%H:%M:%S')}")
        
        buffer.add_5s_bar(
            timestamp=bar_time,
            open_=100.0, high=101.0, low=99.0, close=100.5, volume=1000
        )
    
    # Now trigger next minute to finalize the 14:39 candle
    next_minute = base_time + timedelta(minutes=1)  # 14:40:00
    print(f"  📈 5s bar at {next_minute.strftime('%H:%M:%S')} (next minute)")
    
    buffer.add_5s_bar(
        timestamp=next_minute,
        open_=100.5, high=101.5, low=99.5, close=101.0, volume=1000
    )
    
    # Check results
    df_1m, df_5m, df_4h = buffer.get_dataframes()
    
    print("\n✅ Results:")
    if not df_1m.empty:
        latest_1m = df_1m.iloc[-1]
        print(f"  1m candle timestamp: {latest_1m['date']}")
        print(f"  Expected: 2024-01-15 14:39:00 (period START)")
        
        # Verify timing
        expected = datetime(2024, 1, 15, 14, 39, 0)
        actual = latest_1m['date'].to_pydatetime()
        
        if actual == expected:
            print("  ✅ 1m timing CORRECT")
        else:
            print(f"  ❌ 1m timing WRONG: got {actual}, expected {expected}")
    
    print("\n🎯 Key Point: Candle timestamp = period START time")
    print("  - 14:39 candle covers 14:39:00 to 14:39:59")
    print("  - 14:40 5m candle covers 14:40:00 to 14:44:59") 
    print("  - This ensures indicators calculate correctly!")

if __name__ == "__main__":
    test_candle_timing()