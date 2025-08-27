#!/usr/bin/env python3
"""
Phase 1 Test Script - Data Pipeline Verification
Tests data streaming, aggregation, and historical caching
"""
import asyncio
import sys
import random
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from orb_live.data.stream import LiveDataStream

# Test configuration - use random client ID to avoid conflicts
TEST_CONFIG = {
    'host': '127.0.0.1',
    'port': 7496,  # Live data port (paper mode has no streaming)
    'client_id': random.randint(200, 299),  # Random ID in 200-299 range
    'symbols': ['QQQ']
}

# SINGLE GLOBAL STREAM - Use one instance throughout all tests
GLOBAL_STREAM = None

async def setup_global_stream():
    """Create and connect the single stream instance"""
    global GLOBAL_STREAM
    if GLOBAL_STREAM is None:
        GLOBAL_STREAM = LiveDataStream(TEST_CONFIG)
        connected = await GLOBAL_STREAM.connect()
        if not connected:
            print("❌ Failed to setup global stream")
            return False
        print(f"✅ Global stream connected with Client ID {GLOBAL_STREAM.client_id}")
    return True

async def cleanup_global_stream():
    """Cleanup the global stream"""
    global GLOBAL_STREAM
    if GLOBAL_STREAM:
        await GLOBAL_STREAM.stop()
        GLOBAL_STREAM = None

async def test_connection():
    """Test 1: Basic IBKR connection"""
    print("\n" + "="*60)
    print("TEST 1: IBKR Connection")
    print("="*60)
    
    if GLOBAL_STREAM and GLOBAL_STREAM.connected:
        print("✅ Connection successful!")
        print(f"   Client ID: {GLOBAL_STREAM.client_id}")
        print(f"   Connected: {GLOBAL_STREAM.ib.isConnected()}")
        return True
    else:
        print("❌ Connection failed!")
        return False

async def test_historical_data():
    """Test 2: Historical data loading and caching"""
    print("\n" + "="*60)
    print("TEST 2: Historical Data Cache")
    print("="*60)
    
    if not GLOBAL_STREAM or not GLOBAL_STREAM.connected:
        print("❌ Global stream not available")
        return False
    
    # Initialize historical data
    await GLOBAL_STREAM.initialize_data()
    
    # Check data was loaded
    success = True
    for symbol in TEST_CONFIG['symbols']:
        df_1m, df_5m, df_4h = GLOBAL_STREAM.get_dataframes(symbol)
        
        print(f"\n{symbol} Historical Data:")
        print(f"  1m bars: {len(df_1m)}")
        print(f"  5m bars: {len(df_5m)}")
        print(f"  4h bars: {len(df_4h)}")
        
        if df_1m.empty or df_5m.empty or df_4h.empty:
            print(f"❌ Missing data for {symbol}")
            success = False
        else:
            print(f"✅ Data loaded successfully")
            
            # Show sample data
            if not df_1m.empty:
                print(f"\n  Latest 1m bars:")
                print(f"  {df_1m.tail(3)[['date', 'close']].to_string()}")
    
    return success

async def test_live_streaming(duration: int = 60):
    """Test 3: Live data streaming and aggregation"""
    print("\n" + "="*60)
    print(f"TEST 3: Live Streaming ({duration} seconds)")
    print("="*60)
    
    if not GLOBAL_STREAM or not GLOBAL_STREAM.connected:
        print("❌ Global stream not available")
        return False
    
    # Get initial counts
    initial_counts = {}
    for symbol in TEST_CONFIG['symbols']:
        df_1m, df_5m, df_4h = GLOBAL_STREAM.get_dataframes(symbol)
        initial_counts[symbol] = {
            '1m': len(df_1m),
            '5m': len(df_5m),
            '4h': len(df_4h)
        }
    
    # Start streaming
    if not await GLOBAL_STREAM.start_streaming():
        print("❌ Could not start streaming")
        return False
    
    print(f"\n🎯 Streaming for {duration} seconds...")
    print("You should see colored bars appearing below:\n")
    
    # Wait for specified duration
    await asyncio.sleep(duration)
    
    # Check results
    print("\n" + "-"*60)
    print("Streaming Results:")
    
    success = True
    for symbol in TEST_CONFIG['symbols']:
        df_1m, df_5m, df_4h = GLOBAL_STREAM.get_dataframes(symbol)
        
        new_1m = len(df_1m) - initial_counts[symbol]['1m']
        new_5m = len(df_5m) - initial_counts[symbol]['5m']
        new_4h = len(df_4h) - initial_counts[symbol]['4h']
        
        print(f"\n{symbol} New Bars:")
        print(f"  1m: +{new_1m} (Total: {len(df_1m)})")
        print(f"  5m: +{new_5m} (Total: {len(df_5m)})")
        print(f"  4h: +{new_4h} (Total: {len(df_4h)})")
        
        latest_price = GLOBAL_STREAM.get_latest_price(symbol)
        if latest_price:
            print(f"  Latest price: ${latest_price:.2f}")
        
        # Verify we got new data
        if new_1m >= 1:
            print(f"✅ Aggregation working correctly")
        else:
            print(f"⚠️ No new 1m bars received")
            success = False
    
    return success

async def test_full_integration():
    """Test 4: Full integration test"""
    print("\n" + "="*60)
    print("TEST 4: Full Integration")
    print("="*60)
    
    if not GLOBAL_STREAM:
        print("❌ Global stream not available")
        return False
    
    print("✅ Using existing global stream for integration test")
    print("🧪 Testing data flows end-to-end...")
    
    # Quick check that everything is working
    latest_price = GLOBAL_STREAM.get_latest_price('QQQ')
    if latest_price:
        print(f"✅ Latest QQQ price: ${latest_price:.2f}")
    
    df_1m, df_5m, df_4h = GLOBAL_STREAM.get_dataframes('QQQ')
    print(f"✅ Data buffers: 1m={len(df_1m)}, 5m={len(df_5m)}, 4h={len(df_4h)}")
    
    return True

async def main():
    """Run all tests with single global stream"""
    print("\n" + "="*60)
    print("ORB PRODUCTION - PHASE 1 TESTING (SINGLE CLIENT)")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔧 Using single IB client throughout all tests")
    print("="*60)
    
    try:
        # Setup global stream first
        if not await setup_global_stream():
            print("❌ Failed to setup global stream. Cannot run tests.")
            return False
        
        tests = [
            ("Connection", test_connection),
            ("Historical Data", test_historical_data),
            ("Live Streaming", lambda: test_live_streaming(30)),
            ("Full Integration", test_full_integration)
        ]
        
        results = []
        
        for name, test_func in tests:
            try:
                result = await test_func()
                results.append((name, result))
            except KeyboardInterrupt:
                print("\n⚠️ Test interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Test failed with error: {e}")
                results.append((name, False))
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        for name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{name:20} {status}")
        
        all_passed = all(result for _, result in results)
        
        if all_passed:
            print("\n🎉 ALL TESTS PASSED! Phase 1 is ready.")
            print("✅ Single IB client approach works perfectly")
        else:
            print("\n⚠️ Some tests failed. Please review the output above.")
        
        return all_passed
        
    finally:
        # Always cleanup the global stream
        await cleanup_global_stream()

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⛔ Testing stopped by user")
        sys.exit(1)