#!/usr/bin/env python3
"""
Enhanced Entry Signal Analysis - Marry signals with confidence scores
Debug confidence scoring vs backtesting mismatches
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
    'client_id': 260,  # Different from other tests
    'symbols': ['QQQ']
}

async def run_detailed_entry_analysis():
    """Run detailed entry signal and confidence analysis"""
    print("🔬 DETAILED ENTRY SIGNAL + CONFIDENCE ANALYSIS")
    print("=" * 60)
    
    # Setup data stream
    stream = LiveDataStream(TEST_CONFIG)
    
    try:
        if not await stream.connect():
            print("❌ Failed to connect to IBKR")
            return
        
        await stream.initialize_data()
        
        # Get data
        df_1m, df_5m, df_4h = stream.get_dataframes('QQQ')
        
        if df_1m.empty or df_5m.empty or df_4h.empty:
            print("❌ Insufficient data")
            return
        
        # Add date_only column if missing (the ORBIndicator should do this, but let's be safe)
        for df in [df_1m, df_5m, df_4h]:
            if 'date_only' not in df.columns:
                if hasattr(df.index, 'date'):
                    df['date_only'] = df.index.date
                elif 'date' in df.columns:
                    df['date_only'] = pd.to_datetime(df['date']).dt.date
                else:
                    print(f"⚠️ Warning: Cannot create date_only column - no date info found")
                    print(f"   Index type: {type(df.index)}")
                    print(f"   Columns: {list(df.columns)}")
                    return
        
        # Create ORB system with YAML config
        config = ORBConfig.from_yaml()
        indicator = ORBIndicator(df_1m, df_5m, df_4h, config)
        
        print(f"📊 Data loaded:")
        print(f"   1m bars: {len(df_1m)}")
        print(f"   5m bars: {len(df_5m)}")
        print(f"   4h bars: {len(df_4h)}")
        print(f"   Config: ORB={config.orb_minutes}min, ATR={config.atr_multiplier}")
        
        # Analyze last 2 trading days in detail
        dates = df_1m['date_only'].unique()[-2:]
        
        all_results = []
        
        for date in dates:
            print(f"\n{'='*50}")
            print(f"📅 ANALYZING {date}")
            print(f"{'='*50}")
            
            market_open = pd.Timestamp(datetime.combine(date, time(9, 30)))
            orb_end = market_open + timedelta(minutes=config.orb_minutes)
            entry_window_end = market_open + timedelta(minutes=60)
            
            # Get ORB levels
            orb_levels = indicator.calculate_orb(pd.Timestamp(date))
            if pd.isna(orb_levels['orb_high']):
                print(f"   ⚠️ No ORB data for {date}")
                continue
            
            print(f"🎯 ORB LEVELS:")
            print(f"   High: ${orb_levels['orb_high']:.2f}")
            print(f"   Low: ${orb_levels['orb_low']:.2f}")
            print(f"   Range: ${orb_levels['orb_range']:.2f}")
            
            # Get entry window bars - handle different index types
            if hasattr(df_1m.index, 'date'):
                # DatetimeIndex
                entry_window_bars = df_1m.index[
                    (df_1m.index >= orb_end) & 
                    (df_1m.index <= entry_window_end)
                ]
            elif 'date' in df_1m.columns:
                # Use date column
                df_day = df_1m[df_1m['date_only'] == date]
                date_times = pd.to_datetime(df_day['date'])
                entry_mask = (date_times >= orb_end) & (date_times <= entry_window_end)
                entry_window_bars = df_day.index[entry_mask]
            else:
                print(f"   ⚠️ Cannot filter entry window - unclear data structure")
                continue
            
            print(f"   Entry Window: {orb_end.strftime('%H:%M')} - {entry_window_end.strftime('%H:%M')} ({len(entry_window_bars)} bars)")
            
            # Sample bars for analysis (every 5 minutes)
            sample_bars = entry_window_bars[::5] if len(entry_window_bars) > 0 else []
            
            for timestamp in sample_bars[:8]:  # Max 8 samples per day
                minutes_after_open = (timestamp - market_open).total_seconds() / 60
                current_1m = df_1m.loc[timestamp]
                
                print(f"\n⏰ {timestamp.strftime('%H:%M')} (+{minutes_after_open:.0f}min)")
                print(f"   OHLC: {current_1m['open']:.2f}/{current_1m['high']:.2f}/{current_1m['low']:.2f}/{current_1m['close']:.2f}")
                print(f"   Vol: {current_1m['volume']:,.0f} | RSI: {current_1m.get('rsi', 0):.1f}")
                
                # Analyze both directions
                for direction in [TradeDirection.LONG, TradeDirection.SHORT]:
                    # Entry signal check
                    signal = indicator.check_entry_signal(timestamp, direction)
                    
                    # Confidence components
                    total_score, conf_level = indicator.calculate_confidence_score(timestamp, direction)
                    base_5m = indicator._calculate_5m_confidence_score(timestamp, direction)
                    confluence_4h = indicator._calculate_4h_confluence_score(timestamp, direction)
                    
                    print(f"   {direction.value}: {'✅' if signal else '❌'} | Conf: {total_score:.1f} ({conf_level.value})")
                    print(f"      5M: {base_5m:.1f} | 4H: {confluence_4h:.1f}")
                    
                    # Breakout analysis
                    if direction == TradeDirection.LONG:
                        high_breaks = current_1m['high'] > orb_levels['orb_high']
                        close_confirms = current_1m['close'] > orb_levels['orb_high']
                        breakout_valid = high_breaks and close_confirms
                        print(f"      Breakout: H>{orb_levels['orb_high']:.2f}?{high_breaks} & C>{orb_levels['orb_high']:.2f}?{close_confirms} = {breakout_valid}")
                    else:
                        low_breaks = current_1m['low'] < orb_levels['orb_low']
                        close_confirms = current_1m['close'] < orb_levels['orb_low']
                        breakout_valid = low_breaks and close_confirms
                        print(f"      Breakout: L<{orb_levels['orb_low']:.2f}?{low_breaks} & C<{orb_levels['orb_low']:.2f}?{close_confirms} = {breakout_valid}")
                    
                    # Detailed indicator breakdown
                    indicators_5m = indicator.get_5m_indicators_at_time(timestamp)
                    indicators_4h = indicator.get_4h_indicators_at_time(timestamp)
                    
                    if indicators_5m and (signal or total_score > 4.0):  # Show details for signals or high confidence
                        print(f"      📊 5M: RSI={indicators_5m['rsi_5m']:.1f} MACD_H={indicators_5m['macd_histogram_5m']:.3f}")
                        vol_ratio = indicators_5m['volume_5m'] / indicators_5m['volume_ma_5m']
                        vwap_dist = (indicators_5m['close_5m'] - indicators_5m['vwap_5m']) / indicators_5m['vwap_5m'] * 100
                        print(f"          Vol_Ratio={vol_ratio:.1f} VWAP_Dist={vwap_dist:.2f}%")
                    
                    if indicators_4h and (signal or total_score > 4.0):
                        rsi_slope = indicator._get_rsi_slope_4h(timestamp)
                        ma_dist = ((indicators_4h['close_4h'] - indicators_4h['ema_20_4h']) / indicators_4h['ema_20_4h']) * 100
                        print(f"      📊 4H: RSI={indicators_4h['rsi_4h']:.1f} Slope={rsi_slope:.1f} MA_Dist={ma_dist:.1f}%")
                        print(f"          MACD_H={indicators_4h['macd_histogram_4h']:.3f}")
                    
                    # Store result
                    all_results.append({
                        'date': date,
                        'timestamp': timestamp,
                        'direction': direction.value,
                        'signal': signal,
                        'total_score': total_score,
                        'confidence_level': conf_level.value,
                        '5m_score': base_5m,
                        '4h_score': confluence_4h,
                        'minutes_after_open': minutes_after_open
                    })
        
        # Summary Analysis
        print(f"\n{'='*60}")
        print(f"📈 COMPREHENSIVE SUMMARY")
        print(f"{'='*60}")
        
        if all_results:
            df_results = pd.DataFrame(all_results)
            
            total_tests = len(df_results)
            total_signals = len(df_results[df_results['signal']])
            high_conf_signals = len(df_results[(df_results['signal']) & (df_results['confidence_level'] == 'HIGH')])
            medium_conf_signals = len(df_results[(df_results['signal']) & (df_results['confidence_level'] == 'MEDIUM')])
            
            print(f"🎯 SIGNAL STATISTICS:")
            print(f"   Total Tests: {total_tests}")
            print(f"   Total Signals: {total_signals} ({total_signals/total_tests*100:.1f}%)")
            print(f"   HIGH Confidence: {high_conf_signals}")
            print(f"   MEDIUM Confidence: {medium_conf_signals}")
            
            avg_all_confidence = df_results['total_score'].mean()
            avg_signal_confidence = df_results[df_results['signal']]['total_score'].mean() if total_signals > 0 else 0
            
            print(f"   Average Confidence (All): {avg_all_confidence:.1f}")
            print(f"   Average Confidence (Signals): {avg_signal_confidence:.1f}")
            
            # Score distribution
            print(f"\n📊 CONFIDENCE DISTRIBUTION:")
            for level in ['LOW', 'MEDIUM', 'HIGH']:
                count = len(df_results[df_results['confidence_level'] == level])
                signals = len(df_results[(df_results['confidence_level'] == level) & (df_results['signal'])])
                print(f"   {level}: {count} tests, {signals} signals ({signals/max(count,1)*100:.1f}% signal rate)")
            
            # Component score analysis
            print(f"\n🔍 COMPONENT SCORES:")
            print(f"   5M Score - Avg: {df_results['5m_score'].mean():.1f} | Max: {df_results['5m_score'].max():.1f}")
            print(f"   4H Score - Avg: {df_results['4h_score'].mean():.1f} | Max: {df_results['4h_score'].max():.1f}")
            
            # Best signals
            best_signals = df_results[(df_results['signal']) & (df_results['total_score'] >= 6.0)]
            if not best_signals.empty:
                print(f"\n🏆 BEST SIGNALS (Score ≥ 6.0): {len(best_signals)}")
                for _, row in best_signals.iterrows():
                    print(f"   {row['date']} {row['timestamp'].strftime('%H:%M')} {row['direction']} - Score: {row['total_score']:.1f}")
            
            # Show potential issues
            high_conf_no_signal = df_results[(~df_results['signal']) & (df_results['total_score'] >= 6.0)]
            if not high_conf_no_signal.empty:
                print(f"\n⚠️ HIGH CONFIDENCE BUT NO SIGNAL: {len(high_conf_no_signal)}")
                print("   This might indicate a mismatch in logic!")
                for _, row in high_conf_no_signal.head(3).iterrows():
                    print(f"   {row['date']} {row['timestamp'].strftime('%H:%M')} {row['direction']} - Score: {row['total_score']:.1f}")
        
        print(f"\n✅ Analysis complete!")
        
    finally:
        await stream.stop()

if __name__ == "__main__":
    try:
        asyncio.run(run_detailed_entry_analysis())
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        sys.exit(1)