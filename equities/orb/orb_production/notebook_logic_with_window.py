#!/usr/bin/env python3
"""
Exact Notebook Logic + 60-Minute Entry Window
Combines your proven notebook approach with the 60-minute window constraint
"""
import asyncio
import sys
import pandas as pd
import numpy as np
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
    'client_id': 280,
    'symbols': ['QQQ']
}

async def run_notebook_logic_with_window():
    """Run exact notebook logic but with 60-minute entry window"""
    print("📖 NOTEBOOK LOGIC + 60-MINUTE WINDOW")
    print("=" * 60)
    
    # Setup
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
        print(f"   ORB minutes: {config.orb_minutes}")
        print(f"   Entry window: 60 minutes after market open")
        print(f"   1m bars: {len(indicator.df_1m)}")
        print(f"   5m bars: {len(indicator.df_5m)}")
        print(f"   4h bars: {len(indicator.df_4h)}")
        
        # Check data coverage for confidence scoring
        print(f"\n🔍 DATA COVERAGE ANALYSIS:")
        
        # 4H data coverage
        if len(indicator.df_4h) > 0:
            first_4h = indicator.df_4h.index[0]
            last_4h = indicator.df_4h.index[-1]
            days_of_4h_data = (last_4h - first_4h).days
            print(f"   4H data span: {days_of_4h_data} days ({first_4h.strftime('%Y-%m-%d')} to {last_4h.strftime('%Y-%m-%d')})")
            
            # Check if we have enough for indicators
            valid_rsi_4h = indicator.df_4h['rsi'].notna().sum()
            valid_macd_4h = indicator.df_4h['macd'].notna().sum()
            valid_ema_4h = indicator.df_4h.get('ema_20', pd.Series()).notna().sum()
            
            print(f"   4H indicators valid bars:")
            print(f"     RSI: {valid_rsi_4h}/{len(indicator.df_4h)} ({valid_rsi_4h/len(indicator.df_4h)*100:.1f}%)")
            print(f"     MACD: {valid_macd_4h}/{len(indicator.df_4h)} ({valid_macd_4h/len(indicator.df_4h)*100:.1f}%)")
            print(f"     EMA-20: {valid_ema_4h}/{len(indicator.df_4h)} ({valid_ema_4h/max(len(indicator.df_4h),1)*100:.1f}%)")
            
            if valid_rsi_4h < 14:
                print(f"   ⚠️ WARNING: Only {valid_rsi_4h} valid RSI bars - need 14+ for reliable RSI")
            if valid_macd_4h < 26:
                print(f"   ⚠️ WARNING: Only {valid_macd_4h} valid MACD bars - need 26+ for reliable MACD")
        
        # 5M data coverage - check only calculated indicators
        if len(indicator.df_5m) > 0:
            available_5m_indicators = []
            if 'rsi' in indicator.df_5m.columns:
                valid_rsi_5m = indicator.df_5m['rsi'].notna().sum()
                available_5m_indicators.append(f"RSI: {valid_rsi_5m}/{len(indicator.df_5m)} ({valid_rsi_5m/len(indicator.df_5m)*100:.1f}%)")
            if 'macd' in indicator.df_5m.columns:
                valid_macd_5m = indicator.df_5m['macd'].notna().sum()
                available_5m_indicators.append(f"MACD: {valid_macd_5m}/{len(indicator.df_5m)} ({valid_macd_5m/len(indicator.df_5m)*100:.1f}%)")
            if 'atr' in indicator.df_5m.columns:
                valid_atr_5m = indicator.df_5m['atr'].notna().sum()
                available_5m_indicators.append(f"ATR: {valid_atr_5m}/{len(indicator.df_5m)} ({valid_atr_5m/len(indicator.df_5m)*100:.1f}%)")
            if 'ema_21' in indicator.df_5m.columns:
                valid_ema21_5m = indicator.df_5m['ema_21'].notna().sum()
                available_5m_indicators.append(f"EMA-21: {valid_ema21_5m}/{len(indicator.df_5m)} ({valid_ema21_5m/len(indicator.df_5m)*100:.1f}%)")
            
            print(f"   5M indicators available: {len(available_5m_indicators)} calculated")
            for indicator_info in available_5m_indicators:
                print(f"     {indicator_info}")
            
            if len(available_5m_indicators) < 4:
                print(f"   ⚠️ NOTICE: Limited 5M indicators - will use 1M fallback scoring (matches backtesting)")
        
        # 1M data coverage  
        if len(indicator.df_1m) > 0:
            valid_rsi_1m = indicator.df_1m['rsi'].notna().sum()
            valid_vwap_1m = indicator.df_1m['vwap'].notna().sum()
            print(f"   1M indicators valid bars:")
            print(f"     RSI: {valid_rsi_1m}/{len(indicator.df_1m)} ({valid_rsi_1m/len(indicator.df_1m)*100:.1f}%)")
            print(f"     VWAP: {valid_vwap_1m}/{len(indicator.df_1m)} ({valid_vwap_1m/len(indicator.df_1m)*100:.1f}%)")
        
        print(f"\n💡 For reliable confidence scores, you need:")
        print(f"   • 4H RSI: 14+ bars (14 * 4h = 56+ hours = 7+ trading days)")
        print(f"   • 4H MACD: 26+ bars (26 * 4h = 104+ hours = 13+ trading days)")
        print(f"   • 4H EMA-20: 20+ bars (20 * 4h = 80+ hours = 10+ trading days)")
        
        # EXACT NOTEBOOK LOGIC WITH WINDOW CONSTRAINT
        print("\nANALYZING ALL ORB SIGNALS (WITH 60-MIN WINDOW)")
        print("=" * 70)
        
        # Get all unique trading days
        unique_dates = indicator.df_1m['date_only'].unique()
        trading_dates = sorted([d for d in unique_dates if pd.Timestamp(d).weekday() < 5])
        
        print(f"Total trading days: {len(trading_dates)}")
        print(f"Date range: {trading_dates[0]} to {trading_dates[-1]}")
        print()
        
        # Store all signals
        all_signals = []
        
        # Analyze each trading day
        for trade_date in trading_dates[-5:]:  # Last 5 days for testing
            # Calculate ORB levels for this date
            date_ts = pd.Timestamp(trade_date)
            orb_levels = indicator.calculate_orb(date_ts)
            
            if np.isnan(orb_levels['orb_high']):
                print(f"{trade_date}: No ORB levels (insufficient data)")
                continue
            
            # Get data for this day
            day_data = indicator.df_1m[indicator.df_1m['date_only'] == trade_date]
            
            # Define critical times
            market_open = pd.Timestamp(datetime.combine(trade_date, config.market_open))
            orb_end = market_open + pd.Timedelta(minutes=config.orb_minutes)
            entry_window_end = market_open + pd.Timedelta(minutes=60)  # 60-MINUTE WINDOW
            market_close = pd.Timestamp(datetime.combine(trade_date, config.market_close))
            
            # Get post-ORB data WITHIN 60-MINUTE WINDOW
            post_orb_data = day_data[
                (day_data.index > orb_end) & 
                (day_data.index <= entry_window_end)  # WINDOW CONSTRAINT
            ]
            
            print(f"\n{trade_date}:")
            print(f"   ORB: ${orb_levels['orb_low']:.2f} - ${orb_levels['orb_high']:.2f} (${orb_levels['orb_range']:.2f})")
            print(f"   Entry Window: {orb_end.strftime('%H:%M')} - {entry_window_end.strftime('%H:%M')} ({len(post_orb_data)} bars)")
            
            # ONLY ONE SIGNAL PER DAY - first valid breakout wins
            signal_found = False
            
            # Look for FIRST breakout after ORB completes (EXACT NOTEBOOK LOGIC)
            for timestamp in post_orb_data.index:
                if signal_found:
                    break  # Already found today's signal
                    
                row = post_orb_data.loc[timestamp]
                
                # Check for long signal (break above ORB high) - EXACT NOTEBOOK CONDITIONS
                if row['high'] > orb_levels['orb_high'] and row['close'] > orb_levels['orb_high']:
                    # Calculate confidence score
                    confidence_score, confidence_level = indicator.calculate_confidence_score(timestamp, TradeDirection.LONG)
                    
                    signal = {
                        'date': trade_date,
                        'time': timestamp,
                        'type': 'LONG',
                        'trigger_level': orb_levels['orb_high'],
                        'entry_price': row['close'],  # 1m close price
                        'orb_high': orb_levels['orb_high'],
                        'orb_low': orb_levels['orb_low'],
                        'orb_range': orb_levels['orb_range'],
                        'volume': row['volume'],
                        'rsi': row.get('rsi', np.nan),
                        'confidence_score': confidence_score,
                        'confidence_level': confidence_level.value,
                        'breakout_pct': ((row['close'] - orb_levels['orb_high']) / orb_levels['orb_range'] * 100),
                        'minutes_after_open': (timestamp - market_open).total_seconds() / 60
                    }
                    all_signals.append(signal)
                    signal_found = True
                    
                    print(f"   ✅ LONG signal @ {timestamp.strftime('%H:%M')} (+{signal['minutes_after_open']:.0f}min)")
                    print(f"      Entry: ${row['close']:.2f} | Confidence: {confidence_level.value} ({confidence_score:.1f})")
                    print(f"      Breakout: H={row['high']:.2f} > ORB={orb_levels['orb_high']:.2f}? YES")
                    print(f"      Confirm: C={row['close']:.2f} > ORB={orb_levels['orb_high']:.2f}? YES")
                
                # Check for short signal (break below ORB low) - only if no long signal yet
                elif row['low'] < orb_levels['orb_low'] and row['close'] < orb_levels['orb_low']:
                    # Calculate confidence score
                    confidence_score, confidence_level = indicator.calculate_confidence_score(timestamp, TradeDirection.SHORT)
                    
                    signal = {
                        'date': trade_date,
                        'time': timestamp,
                        'type': 'SHORT',
                        'trigger_level': orb_levels['orb_low'],
                        'entry_price': row['close'],  # 1m close price
                        'orb_high': orb_levels['orb_high'],
                        'orb_low': orb_levels['orb_low'],
                        'orb_range': orb_levels['orb_range'],
                        'volume': row['volume'],
                        'rsi': row.get('rsi', np.nan),
                        'confidence_score': confidence_score,
                        'confidence_level': confidence_level.value,
                        'breakout_pct': ((orb_levels['orb_low'] - row['close']) / orb_levels['orb_range'] * 100),
                        'minutes_after_open': (timestamp - market_open).total_seconds() / 60
                    }
                    all_signals.append(signal)
                    signal_found = True
                    
                    print(f"   ✅ SHORT signal @ {timestamp.strftime('%H:%M')} (+{signal['minutes_after_open']:.0f}min)")
                    print(f"      Entry: ${row['close']:.2f} | Confidence: {confidence_level.value} ({confidence_score:.1f})")
                    print(f"      Breakout: L={row['low']:.2f} < ORB={orb_levels['orb_low']:.2f}? YES")
                    print(f"      Confirm: C={row['close']:.2f} < ORB={orb_levels['orb_low']:.2f}? YES")
            
            # Report if no signal found
            if not signal_found:
                print(f"   ❌ No breakouts within 60-minute window")
        
        # Create DataFrame of all signals
        print(f"\n{'='*70}")
        print(f"📈 FINAL RESULTS")
        print(f"{'='*70}")
        
        if all_signals:
            signals_df = pd.DataFrame(all_signals)
            
            print(f"📊 TOTAL SIGNALS FOUND: {len(signals_df)}")
            print(f"   Long signals: {len(signals_df[signals_df['type'] == 'LONG'])}")
            print(f"   Short signals: {len(signals_df[signals_df['type'] == 'SHORT'])}")
            
            # Confidence analysis
            for level in ['HIGH', 'MEDIUM', 'LOW']:
                count = len(signals_df[signals_df['confidence_level'] == level])
                if count > 0:
                    avg_score = signals_df[signals_df['confidence_level'] == level]['confidence_score'].mean()
                    print(f"   {level} confidence: {count} signals (avg score: {avg_score:.1f})")
            
            # Entry timing analysis
            avg_entry_time = signals_df['minutes_after_open'].mean()
            print(f"\n⏱️ TIMING ANALYSIS:")
            print(f"   Average entry time: {avg_entry_time:.1f} minutes after open")
            print(f"   Entry time range: {signals_df['minutes_after_open'].min():.0f} - {signals_df['minutes_after_open'].max():.0f} minutes")
            
            # Show all signals
            print(f"\n📋 ALL SIGNALS:")
            for _, signal in signals_df.iterrows():
                print(f"   {signal['date']} {signal['time'].strftime('%H:%M')} {signal['type']} - "
                      f"Entry: ${signal['entry_price']:.2f} | Conf: {signal['confidence_level']} ({signal['confidence_score']:.1f})")
                
        else:
            print("❌ No signals found in the dataset")
            print("   This might indicate:")
            print("   1. 60-minute window is too restrictive")
            print("   2. ORB breakouts happen later in the day")
            print("   3. Market conditions don't produce breakouts in this period")
        
        print(f"\n✅ Analysis complete!")
        
    finally:
        await stream.stop()

if __name__ == "__main__":
    try:
        asyncio.run(run_notebook_logic_with_window())
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()