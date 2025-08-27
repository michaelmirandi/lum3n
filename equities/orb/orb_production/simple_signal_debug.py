#!/usr/bin/env python3
"""
Simple Signal Debug - Use existing test framework to debug confidence vs signal mismatch
"""
import asyncio
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime, time, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from orb_live.data.stream import LiveDataStream
from orb_live.core.orb_system import ORBIndicator, ORBConfig, TradeDirection, ConfidenceLevel

# Test configuration
TEST_CONFIG = {
    'host': '127.0.0.1',
    'port': 7496,
    'client_id': 270,
    'symbols': ['QQQ']
}

async def debug_signals():
    """Debug signals using the working test framework approach"""
    print("🔬 SIMPLE SIGNAL DEBUG")
    print("=" * 40)
    
    # Setup like the working test_phase2.py
    stream = LiveDataStream(TEST_CONFIG)
    
    try:
        if not await stream.connect():
            print("❌ Failed to connect to IBKR")
            return
        
        await stream.initialize_data()
        df_1m, df_5m, df_4h = stream.get_dataframes('QQQ')
        
        if df_1m.empty or df_5m.empty or df_4h.empty:
            print("❌ Insufficient data")
            return
        
        # Create ORB system with YAML config
        config = ORBConfig.from_yaml()
        indicator = ORBIndicator(df_1m, df_5m, df_4h, config)
        
        print(f"📊 System ready:")
        print(f"   1m bars: {len(df_1m)}")
        print(f"   5m bars: {len(df_5m)}")  
        print(f"   4h bars: {len(df_4h)}")
        print(f"   ORB minutes: {config.orb_minutes}")
        print(f"   High conf threshold: {config.confidence_high_threshold}")
        
        # Use the existing indicator dataframes (already prepared)
        df_1m_ready = indicator.df_1m
        df_5m_ready = indicator.df_5m
        df_4h_ready = indicator.df_4h
        
        print(f"\n🔍 After indicator preparation:")
        print(f"   1m ready: {len(df_1m_ready)} bars")
        print(f"   1m index type: {type(df_1m_ready.index)}")
        print(f"   1m columns: {list(df_1m_ready.columns)}")
        
        # Get recent trading dates
        if 'date_only' in df_1m_ready.columns:
            recent_dates = df_1m_ready['date_only'].unique()[-2:]
            print(f"   Recent dates: {recent_dates}")
        else:
            print("   ❌ No date_only column found")
            return
        
        # Test entry window detection for recent dates
        results = []
        
        for date in recent_dates:
            print(f"\n📅 Testing {date}")
            
            # Get ORB levels
            orb_levels = indicator.calculate_orb(pd.Timestamp(date))
            if pd.isna(orb_levels['orb_high']):
                print(f"   ❌ No ORB levels for {date}")
                continue
                
            print(f"   ORB: ${orb_levels['orb_low']:.2f} - ${orb_levels['orb_high']:.2f} (${orb_levels['orb_range']:.2f})")
            
            # Find entry window bars for this date
            day_data = df_1m_ready[df_1m_ready['date_only'] == date]
            
            if day_data.empty:
                print(f"   ❌ No 1m data for {date}")
                continue
            
            print(f"   Found {len(day_data)} bars for {date}")
            
            # Get market times
            market_open = pd.Timestamp(datetime.combine(date, time(9, 30)))
            orb_end = market_open + timedelta(minutes=config.orb_minutes)
            entry_window_end = market_open + timedelta(minutes=60)
            
            # Filter to entry window (9:50-10:30 for 20min ORB)
            entry_window_data = day_data[
                (day_data.index >= orb_end) & 
                (day_data.index <= entry_window_end)
            ]
            
            print(f"   Entry window: {len(entry_window_data)} bars ({orb_end.strftime('%H:%M')} - {entry_window_end.strftime('%H:%M')})")
            
            if entry_window_data.empty:
                print(f"   ❌ No entry window data")
                continue
            
            # Sample a few bars for detailed analysis  
            sample_indices = [0, len(entry_window_data)//2, -1] if len(entry_window_data) > 2 else [0]
            
            for idx in sample_indices:
                if idx >= len(entry_window_data):
                    continue
                    
                timestamp = entry_window_data.index[idx]
                bar_data = entry_window_data.iloc[idx]
                
                minutes_after_open = (timestamp - market_open).total_seconds() / 60
                
                print(f"\n   ⏰ {timestamp.strftime('%H:%M')} (+{minutes_after_open:.0f}min)")
                print(f"      OHLC: {bar_data['open']:.2f}/{bar_data['high']:.2f}/{bar_data['low']:.2f}/{bar_data['close']:.2f}")
                
                # Test both directions
                for direction in [TradeDirection.LONG, TradeDirection.SHORT]:
                    # Get signal
                    signal = indicator.check_entry_signal(timestamp, direction)
                    
                    # Get confidence  
                    total_score, conf_level = indicator.calculate_confidence_score(timestamp, direction)
                    
                    # Get components
                    base_5m = indicator._calculate_5m_confidence_score(timestamp, direction)
                    confluence_4h = indicator._calculate_4h_confluence_score(timestamp, direction)
                    
                    print(f"      {direction.value}: {'✅' if signal else '❌'} | Score: {total_score:.1f} ({conf_level.value}) [5M:{base_5m:.1f} + 4H:{confluence_4h:.1f}]")
                    
                    # Check breakout manually
                    if direction == TradeDirection.LONG:
                        high_break = bar_data['high'] > orb_levels['orb_high']
                        close_confirm = bar_data['close'] > orb_levels['orb_high']
                        breakout = high_break and close_confirm
                        print(f"         Breakout: H>{orb_levels['orb_high']:.2f}?{high_break} & C>{orb_levels['orb_high']:.2f}?{close_confirm} = {breakout}")
                    else:
                        low_break = bar_data['low'] < orb_levels['orb_low']
                        close_confirm = bar_data['close'] < orb_levels['orb_low']
                        breakout = low_break and close_confirm
                        print(f"         Breakout: L<{orb_levels['orb_low']:.2f}?{low_break} & C<{orb_levels['orb_low']:.2f}?{close_confirm} = {breakout}")
                    
                    # Store result
                    results.append({
                        'date': date,
                        'time': timestamp.strftime('%H:%M'),
                        'direction': direction.value,
                        'signal': signal,
                        'breakout': breakout,
                        'total_score': total_score,
                        'conf_level': conf_level.value,
                        '5m_score': base_5m,
                        '4h_score': confluence_4h,
                        'signal_matches_breakout': signal == breakout
                    })
        
        # Summary
        print(f"\n{'='*50}")
        print(f"📈 RESULTS SUMMARY")
        print(f"{'='*50}")
        
        if results:
            df_results = pd.DataFrame(results)
            
            total_tests = len(df_results)
            total_signals = len(df_results[df_results['signal']])
            total_breakouts = len(df_results[df_results['breakout']])
            matches = len(df_results[df_results['signal_matches_breakout']])
            
            print(f"📊 Signal vs Breakout Analysis:")
            print(f"   Total tests: {total_tests}")
            print(f"   Signals detected: {total_signals}")
            print(f"   Breakouts detected: {total_breakouts}")
            print(f"   Signal matches breakout: {matches}/{total_tests} ({matches/total_tests*100:.1f}%)")
            
            # Check for mismatches
            mismatches = df_results[~df_results['signal_matches_breakout']]
            if not mismatches.empty:
                print(f"\n⚠️ MISMATCHES FOUND: {len(mismatches)}")
                for _, row in mismatches.iterrows():
                    print(f"   {row['date']} {row['time']} {row['direction']}: Signal={row['signal']}, Breakout={row['breakout']}")
            
            # High confidence analysis
            high_conf = df_results[df_results['conf_level'] == 'HIGH']
            if not high_conf.empty:
                print(f"\n🎯 HIGH CONFIDENCE Analysis: {len(high_conf)} cases")
                high_conf_signals = len(high_conf[high_conf['signal']])
                print(f"   HIGH confidence with signals: {high_conf_signals}/{len(high_conf)}")
                
            # Score distribution
            avg_score = df_results['total_score'].mean()
            avg_5m = df_results['5m_score'].mean()
            avg_4h = df_results['4h_score'].mean()
            
            print(f"\n📊 Score Averages:")
            print(f"   Total: {avg_score:.1f}")
            print(f"   5M: {avg_5m:.1f}")
            print(f"   4H: {avg_4h:.1f}")
            
        print(f"\n✅ Debug analysis complete!")
        
    finally:
        await stream.stop()

if __name__ == "__main__":
    try:
        asyncio.run(debug_signals())
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()