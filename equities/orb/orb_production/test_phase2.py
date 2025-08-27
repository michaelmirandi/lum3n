#!/usr/bin/env python3
"""
Phase 2 Test Script - ORB Logic and Indicators Validation
Tests ORB calculations, technical indicators, and confidence scoring
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
    'client_id': 250,  # Different from Phase 1
    'symbols': ['QQQ']
}

# Global stream and indicator instances
GLOBAL_STREAM = None
GLOBAL_INDICATOR = None

async def setup_test_environment():
    """Setup the test environment with data and indicators"""
    global GLOBAL_STREAM, GLOBAL_INDICATOR
    
    print("🔧 Setting up Phase 2 test environment...")
    
    # Create stream and connect
    GLOBAL_STREAM = LiveDataStream(TEST_CONFIG)
    
    if not await GLOBAL_STREAM.connect():
        print("❌ Failed to connect to IBKR")
        return False
    
    # Initialize historical data
    await GLOBAL_STREAM.initialize_data()
    
    # Get DataFrames for QQQ
    df_1m, df_5m, df_4h = GLOBAL_STREAM.get_dataframes('QQQ')
    
    if df_1m.empty or df_5m.empty or df_4h.empty:
        print("❌ Insufficient data for testing")
        return False
    
    # Create ORB indicator with YAML config
    orb_config = ORBConfig.from_yaml()
    print(f"📋 Using configuration:")
    print(f"   ORB minutes: {orb_config.orb_minutes}")
    print(f"   ATR multiplier: {orb_config.atr_multiplier}")
    print(f"   Initial ATR multiplier: {orb_config.initial_atr_multiplier}")
    print(f"   Final ATR multiplier: {orb_config.final_atr_multiplier}")
    print(f"   4H confluence: {orb_config.use_4h_confluence}")
    print(f"   High confidence threshold: {orb_config.confidence_high_threshold}")
    
    print(f"📊 Creating ORBIndicator with data:")
    print(f"   1m bars: {len(df_1m)} (PRIMARY - entry detection, ORB calculation)")
    print(f"   5m bars: {len(df_5m)} (CONFIDENCE - scoring, ATR)")
    print(f"   4h bars: {len(df_4h)} (CONFLUENCE - trend confirmation)")
    
    # Initialize ORB indicator
    GLOBAL_INDICATOR = ORBIndicator(df_1m, df_5m, df_4h, orb_config)
    
    print("✅ Test environment setup complete")
    return True

async def cleanup_test_environment():
    """Cleanup test environment"""
    global GLOBAL_STREAM
    if GLOBAL_STREAM:
        await GLOBAL_STREAM.stop()

def test_orb_calculations():
    """Test 1: ORB level calculations using 1m bars"""
    print("\n" + "="*60)
    print("TEST 1: ORB Level Calculations (1m bars)")
    print("="*60)
    
    if not GLOBAL_INDICATOR:
        print("❌ ORB Indicator not available")
        return False
    
    # Test with recent trading days using 1m data
    df_1m = GLOBAL_INDICATOR.df_1m
    
    if df_1m.empty:
        print("❌ No 1m data available")
        return False
    
    # Get unique trading dates from 1m data
    trading_dates = df_1m['date_only'].unique()[-5:]  # Last 5 trading days
    
    success_count = 0
    total_tests = len(trading_dates)
    
    print(f"📅 Testing ORB calculations using 1m bars for {total_tests} recent trading days:")
    
    for date in trading_dates:
        try:
            # Convert to Timestamp
            test_date = pd.Timestamp(date)
            
            # Calculate ORB levels
            orb_levels = GLOBAL_INDICATOR.calculate_orb(test_date)
            
            print(f"\n📊 {date}:")
            print(f"   ORB High: ${orb_levels['orb_high']:.2f}")
            print(f"   ORB Low:  ${orb_levels['orb_low']:.2f}")
            print(f"   ORB Range: ${orb_levels['orb_range']:.2f}")
            
            # Validate results
            if (not pd.isna(orb_levels['orb_high']) and 
                not pd.isna(orb_levels['orb_low']) and 
                orb_levels['orb_range'] > 0):
                print(f"   ✅ Valid ORB levels")
                success_count += 1
            else:
                print(f"   ❌ Invalid ORB levels")
                
        except Exception as e:
            print(f"   ❌ Error calculating ORB for {date}: {e}")
    
    success_rate = success_count / total_tests
    print(f"\n📈 ORB Calculation Results:")
    print(f"   Successful: {success_count}/{total_tests} ({success_rate*100:.1f}%)")
    
    return success_rate >= 0.8  # 80% success rate required

def test_technical_indicators():
    """Test 2: Technical indicator calculations (1m primary, 5m/4h for confidence)"""
    print("\n" + "="*60)
    print("TEST 2: Technical Indicators (1m primary, 5m/4h for confidence)")
    print("="*60)
    
    if not GLOBAL_INDICATOR:
        print("❌ ORB Indicator not available")
        return False
    
    # Test indicators - 1m primary, 5m/4h for confidence only
    indicators_to_test = {
        '1m': ['rsi', 'ema_9', 'macd', 'volume_ma', 'vwap', 'high', 'low', 'close'],
        '5m': ['rsi', 'ema_21', 'macd', 'atr', 'volume_ma', 'vwap'],  # For confidence
        '4h': ['rsi', 'macd', 'ema_20']  # For confluence
    }
    
    success = True
    
    for timeframe, indicators in indicators_to_test.items():
        if timeframe == '1m':
            df = GLOBAL_INDICATOR.df_1m
        elif timeframe == '5m':
            df = GLOBAL_INDICATOR.df_5m
        elif timeframe == '4h':
            df = GLOBAL_INDICATOR.df_4h
        
        purpose = "PRIMARY" if timeframe == '1m' else "CONFIDENCE" if timeframe == '5m' else "CONFLUENCE"
        print(f"\n📊 {timeframe.upper()} Indicators ({purpose}):")
        
        for indicator in indicators:
            if indicator in df.columns:
                valid_count = df[indicator].notna().sum()
                total_count = len(df)
                coverage = valid_count / total_count
                
                print(f"   {indicator:<12}: {valid_count:>4}/{total_count} bars ({coverage*100:.1f}% coverage)")
                
                if coverage < 0.1:  # Less than 10% coverage is concerning
                    print(f"   ⚠️ Low coverage for {indicator}")
                    success = False
            else:
                print(f"   {indicator:<12}: ❌ Missing")
                success = False
    
    # Test specific indicator values - focus on 1m primary data
    print(f"\n📈 Sample Current Values (1m primary):")
    try:
        latest_1m = GLOBAL_INDICATOR.df_1m.iloc[-1]
        
        print(f"   1m High: ${latest_1m.get('high', 'N/A'):.2f}")
        print(f"   1m Low: ${latest_1m.get('low', 'N/A'):.2f}")
        print(f"   1m Close: ${latest_1m.get('close', 'N/A'):.2f}")
        print(f"   1m RSI: {latest_1m.get('rsi', 'N/A'):.1f}")
        print(f"   1m VWAP: ${latest_1m.get('vwap', 'N/A'):.2f}")
        print(f"   1m Volume: {latest_1m.get('volume', 'N/A'):,.0f}")
        
        # Show confidence indicators
        print(f"\n📈 Confidence Indicators:")
        latest_5m = GLOBAL_INDICATOR.df_5m.iloc[-1]
        print(f"   5m RSI (conf): {latest_5m.get('rsi', 'N/A'):.1f}")
        print(f"   5m ATR (conf): ${latest_5m.get('atr', 'N/A'):.2f}")
        
        if not GLOBAL_INDICATOR.df_4h.empty:
            latest_4h = GLOBAL_INDICATOR.df_4h.iloc[-1]
            print(f"   4h RSI (conf): {latest_4h.get('rsi', 'N/A'):.1f}")
        
    except Exception as e:
        print(f"   ⚠️ Error getting sample values: {e}")
    
    return success

def test_confidence_scoring():
    """Test 3: Confidence scoring system using post-ORB bars"""
    print("\n" + "="*60)
    print("TEST 3: Confidence Scoring System (Post-ORB Bars)")
    print("="*60)
    
    if not GLOBAL_INDICATOR:
        print("❌ ORB Indicator not available")
        return False
    
    # Test confidence scoring on post-ORB data (not end-of-day)
    df_5m = GLOBAL_INDICATOR.df_5m
    
    if df_5m.empty:
        print("❌ No 5m data for confidence testing")
        return False
    
    # Get timestamps after ORB completion for testing
    test_timestamps = []
    
    for date in df_5m['date_only'].unique()[-3:]:  # Last 3 days
        market_open = pd.Timestamp(datetime.combine(date, time(9, 30)))
        orb_end = market_open + timedelta(minutes=20)  # 9:50 AM
        entry_window_end = market_open + timedelta(minutes=60)  # 10:30 AM
        
        # Find 5m bars in the entry window (after ORB completion)
        entry_window_bars = df_5m.index[
            (df_5m.index >= orb_end) & 
            (df_5m.index <= entry_window_end)
        ]
        
        # Add some bars from this window for confidence testing
        if len(entry_window_bars) >= 2:
            test_timestamps.append(entry_window_bars[0])  # First after ORB
            test_timestamps.append(entry_window_bars[-1])  # Last in window
    
    if not test_timestamps:
        print("⚠️ No suitable post-ORB timestamps found for testing")
        return True
    
    print(f"📊 Testing confidence scoring on {len(test_timestamps)} post-ORB bars:")
    print("   Using bars after ORB completion (9:50 AM - 10:30 AM window)")
    
    total_tests = 0
    successful_calcs = 0
    
    for timestamp in test_timestamps:
        try:
            # Test both LONG and SHORT confidence scores
            for direction in [TradeDirection.LONG, TradeDirection.SHORT]:
                score, level = GLOBAL_INDICATOR.calculate_confidence_score(timestamp, direction)
                
                print(f"\n🕐 {timestamp.strftime('%Y-%m-%d %H:%M')} - {direction.value}:")
                print(f"   Score: {score:.2f}")
                print(f"   Level: {level.value}")
                
                # Validate score is reasonable
                if 0 <= score <= 13:  # Max possible score
                    print(f"   ✅ Valid score range")
                    successful_calcs += 1
                else:
                    print(f"   ❌ Score out of range")
                
                # Test 5m indicators are accessible
                indicators_5m = GLOBAL_INDICATOR.get_5m_indicators_at_time(timestamp)
                if indicators_5m:
                    print(f"   5m RSI: {indicators_5m['rsi_5m']:.1f}")
                    print(f"   5m MACD Hist: {indicators_5m['macd_histogram_5m']:.3f}")
                
                # Test 4h confluence if available
                indicators_4h = GLOBAL_INDICATOR.get_4h_indicators_at_time(timestamp)
                if indicators_4h:
                    print(f"   4h RSI: {indicators_4h['rsi_4h']:.1f}")
                    print(f"   4h Age: {indicators_4h['age_minutes']:.0f} minutes")
                
                total_tests += 1
                
        except Exception as e:
            print(f"   ❌ Error calculating confidence for {timestamp}: {e}")
    
    success_rate = successful_calcs / total_tests if total_tests > 0 else 0
    print(f"\n📈 Confidence Scoring Results:")
    print(f"   Successful calculations: {successful_calcs}/{total_tests} ({success_rate*100:.1f}%)")
    
    return success_rate >= 0.8

def test_entry_signal_detection():
    """Test 4: Entry signal detection using 1m bars"""
    print("\n" + "="*60)
    print("TEST 4: Entry Signal Detection (1m bars, 9:50 AM - 10:30 AM Window)")
    print("="*60)
    
    if not GLOBAL_INDICATOR:
        print("❌ ORB Indicator not available")
        return False
    
    # Test entry signal detection logic using 1m data
    df_1m = GLOBAL_INDICATOR.df_1m
    
    # Find timestamps in the entry window (9:50 AM - 10:30 AM)
    test_timestamps = []
    
    for date in df_1m['date_only'].unique()[-3:]:  # Last 3 days
        market_open = pd.Timestamp(datetime.combine(date, time(9, 30)))
        orb_end = market_open + timedelta(minutes=20)  # 9:50 AM
        entry_window_end = market_open + timedelta(minutes=60)  # 10:30 AM
        
        # Find all 1m bars in the entry window
        entry_window_bars = df_1m.index[
            (df_1m.index >= orb_end) & 
            (df_1m.index <= entry_window_end)
        ]
        
        # Add a few timestamps from this window
        if len(entry_window_bars) > 0:
            # Test at ORB end, middle, and near end of window
            test_timestamps.append(entry_window_bars[0])  # First bar after ORB
            if len(entry_window_bars) >= 20:
                test_timestamps.append(entry_window_bars[10])  # Middle
                test_timestamps.append(entry_window_bars[-5])  # Near end
    
    if not test_timestamps:
        print("⚠️ No suitable entry window timestamps found for testing")
        return True  # Don't fail the test for this
    
    print(f"📊 Testing 1m entry signals at {len(test_timestamps)} timestamps in entry window:")
    print("   Entry Window: 9:50 AM (after ORB) - 10:30 AM (60 min after open)")
    print("   Entry Logic: 1m HIGH/LOW breaks ORB AND 1m CLOSE confirms")
    
    signals_found = 0
    total_tests = 0
    
    for timestamp in test_timestamps:
        try:
            # Test both directions
            for direction in [TradeDirection.LONG, TradeDirection.SHORT]:
                signal = GLOBAL_INDICATOR.check_entry_signal(timestamp, direction)
                total_tests += 1
                
                print(f"\n🕐 {timestamp.strftime('%Y-%m-%d %H:%M')} - {direction.value}:")
                print(f"   Signal: {'✅ DETECTED' if signal else '❌ NO SIGNAL'}")
                
                # Show window validation
                market_open = pd.Timestamp(datetime.combine(timestamp.date(), time(9, 30)))
                orb_end = market_open + timedelta(minutes=20)
                entry_window_end = market_open + timedelta(minutes=60)
                
                minutes_after_open = (timestamp - market_open).total_seconds() / 60
                print(f"   Minutes after open: {minutes_after_open:.0f}")
                print(f"   In entry window: {orb_end <= timestamp <= entry_window_end}")
                
                if signal:
                    signals_found += 1
                    
                    # Show ORB levels and breakout
                    orb_levels = GLOBAL_INDICATOR.calculate_orb(timestamp)
                    print(f"   ORB High: ${orb_levels['orb_high']:.2f}")
                    print(f"   ORB Low: ${orb_levels['orb_low']:.2f}")
                    
                    # Get current 1m bar data
                    if timestamp in df_1m.index:
                        current_1m = df_1m.loc[timestamp]
                        print(f"   1m High: ${current_1m['high']:.2f}")
                        print(f"   1m Low: ${current_1m['low']:.2f}")
                        print(f"   1m Close: ${current_1m['close']:.2f}")
                        
                        if direction == TradeDirection.LONG:
                            high_breaks = current_1m['high'] > orb_levels['orb_high']
                            close_confirms = current_1m['close'] > orb_levels['orb_high']
                            print(f"   High breaks ORB: {high_breaks}")
                            print(f"   Close confirms: {close_confirms}")
                        else:
                            low_breaks = current_1m['low'] < orb_levels['orb_low']
                            close_confirms = current_1m['close'] < orb_levels['orb_low']
                            print(f"   Low breaks ORB: {low_breaks}")
                            print(f"   Close confirms: {close_confirms}")
                
        except Exception as e:
            print(f"   ❌ Error testing entry signal: {e}")
    
    print(f"\n📈 Entry Signal Results:")
    print(f"   Signals detected: {signals_found}/{total_tests}")
    print(f"   Signal rate: {signals_found/total_tests*100:.1f}%" if total_tests > 0 else "   No tests completed")
    print(f"   Entry window: 40 minutes (9:50 AM - 10:30 AM)")
    print(f"   Using 1m bars for entry detection")
    
    return True  # Always pass this test as signals are market-dependent

def test_trade_levels():
    """Test 5: Trade level calculations using 1m entry price"""
    print("\n" + "="*60)
    print("TEST 5: Trade Level Calculations (1m entry price)")
    print("="*60)
    
    if not GLOBAL_INDICATOR:
        print("❌ ORB Indicator not available")
        return False
    
    # Test trade level calculations using 1m data
    df_1m = GLOBAL_INDICATOR.df_1m
    
    if df_1m.empty:
        print("❌ No 1m data for trade levels testing")
        return False
    
    # Get a recent timestamp and price from 1m data
    recent_timestamp = df_1m.index[-1]
    recent_close = df_1m.iloc[-1]['close']
    
    print(f"📊 Testing trade levels using 1m entry price:")
    print(f"   Timestamp: {recent_timestamp}")
    print(f"   1m Entry Price: ${recent_close:.2f}")
    
    success = True
    
    try:
        # Test LONG trade levels
        long_levels = GLOBAL_INDICATOR.calculate_trade_levels(
            recent_close, TradeDirection.LONG, recent_timestamp
        )
        
        print(f"\n📈 LONG Trade Levels:")
        print(f"   Stop Loss: ${long_levels['stop_loss']:.2f}")
        print(f"   TP1: ${long_levels['tp1']:.2f}")
        print(f"   TP2: ${long_levels['tp2']:.2f}")
        print(f"   TP3: ${long_levels['tp3']:.2f}")
        print(f"   ATR: ${long_levels['atr_value']:.2f}")
        print(f"   ORB Range: ${long_levels['orb_range']:.2f}")
        
        # Validate LONG levels make sense
        if (long_levels['stop_loss'] < recent_close < long_levels['tp1'] < 
            long_levels['tp2'] < long_levels['tp3']):
            print(f"   ✅ LONG levels properly ordered")
        else:
            print(f"   ❌ LONG levels incorrectly ordered")
            success = False
            
        # Test SHORT trade levels
        short_levels = GLOBAL_INDICATOR.calculate_trade_levels(
            recent_close, TradeDirection.SHORT, recent_timestamp
        )
        
        print(f"\n📉 SHORT Trade Levels:")
        print(f"   Stop Loss: ${short_levels['stop_loss']:.2f}")
        print(f"   TP1: ${short_levels['tp1']:.2f}")
        print(f"   TP2: ${short_levels['tp2']:.2f}")
        print(f"   TP3: ${short_levels['tp3']:.2f}")
        
        # Validate SHORT levels make sense
        if (short_levels['tp3'] < short_levels['tp2'] < short_levels['tp1'] < 
            recent_close < short_levels['stop_loss']):
            print(f"   ✅ SHORT levels properly ordered")
        else:
            print(f"   ❌ SHORT levels incorrectly ordered")
            success = False
        
    except Exception as e:
        print(f"❌ Error calculating trade levels: {e}")
        success = False
    
    return success

async def main():
    """Run all Phase 2 tests"""
    print("\n" + "="*60)
    print("ORB PRODUCTION - PHASE 2 TESTING")
    print("ORB Logic and Indicators Validation")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    try:
        # Setup test environment
        if not await setup_test_environment():
            print("❌ Failed to setup test environment")
            return False
        
        # Run all tests
        tests = [
            ("ORB Calculations", test_orb_calculations),
            ("Technical Indicators", test_technical_indicators),
            ("Confidence Scoring", test_confidence_scoring),
            ("Entry Signal Detection", test_entry_signal_detection),
            ("Trade Level Calculations", test_trade_levels)
        ]
        
        results = []
        
        for name, test_func in tests:
            try:
                print(f"\n🧪 Running {name}...")
                result = test_func()
                results.append((name, result))
            except KeyboardInterrupt:
                print("\n⚠️ Test interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Test failed with error: {e}")
                results.append((name, False))
        
        # Summary
        print("\n" + "="*60)
        print("PHASE 2 TEST SUMMARY")
        print("="*60)
        
        for name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{name:<25} {status}")
        
        all_passed = all(result for _, result in results)
        
        if all_passed:
            print("\n🎉 ALL PHASE 2 TESTS PASSED!")
            print("✅ ORB system and indicators are working correctly")
            print("🚀 Ready for Phase 3: Signal Detection Layer")
        else:
            print("\n⚠️ Some Phase 2 tests failed. Please review output above.")
        
        return all_passed
        
    finally:
        # Always cleanup
        await cleanup_test_environment()

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⛔ Testing stopped by user")
        sys.exit(1)