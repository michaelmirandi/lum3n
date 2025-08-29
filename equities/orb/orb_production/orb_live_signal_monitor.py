#!/usr/bin/env python3
"""
🚨 ORB LIVE SIGNAL MONITOR - REAL-TIME SIGNAL DETECTION
Monitors QQQ in real-time and alerts on ORB breakouts within 60-minute window

Usage: python orb_live_signal_monitor.py
"""
import asyncio
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime, time, timedelta
import pytz

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from orb_live.data.stream import LiveDataStream
from orb_live.core.orb_system import ORBIndicator, ORBConfig

# Configuration
CONFIG = {
    'host': '127.0.0.1',
    'port': 7496,
    'client_id': 350,  # Unique for live monitor
    'symbols': ['QQQ']
}

# Track signal state
signal_state = {
    'orb_calculated': False,
    'orb_levels': None,
    'signal_triggered': False,
    'signal_details': None,
    'last_check': None
}

async def run_live_signal_monitor():
    """Live ORB Signal Monitor - Real-time detection"""
    print("🚨 ORB LIVE SIGNAL MONITOR")
    print("=" * 60)
    print("⚡ REAL-TIME MODE - Monitoring for breakouts...")
    print("=" * 60)
    
    # Setup data stream
    stream = LiveDataStream(CONFIG)
    
    try:
        # Connect to IBKR
        print("📡 Connecting to IBKR TWS...")
        if not await stream.connect():
            print("❌ Failed to connect - ensure TWS is running on port 7496")
            return
        print("✅ Connected to IBKR")
        
        # Initialize data
        print("💾 Loading historical data for indicators...")
        await stream.initialize_data()
        print("✅ Data initialized")
        
        # Start live streaming
        print("📡 Starting live data stream...")
        await stream.start_streaming()
        print("✅ Live streaming started")
        
        # Create ORB system
        config = ORBConfig.from_yaml()
        
        # Main monitoring loop
        print(f"\n🔴 LIVE MONITORING STARTED")
        print(f"📊 Symbol: QQQ")
        print(f"⏱️  ORB Period: {config.orb_minutes} minutes")
        print(f"🎯 Entry Window: 60 minutes from market open")
        print(f"📈 Waiting for signals...\n")
        print("-" * 60)
        
        check_interval = 5  # Check every 5 seconds
        last_bar_time = None
        
        while True:
            try:
                # CONSISTENT TIMEZONE: Use single Eastern timezone object throughout
                eastern = pytz.timezone('US/Eastern')
                now = datetime.now(eastern)
                current_time = now.time()
                
                # Check if market is open
                if current_time < config.market_open or current_time > config.market_close:
                    if signal_state['last_check'] != 'closed':
                        print(f"\n💤 Market closed - waiting for {config.market_open}...")
                        signal_state['last_check'] = 'closed'
                    await asyncio.sleep(60)  # Check less frequently when market closed
                    continue
                
                # Get latest data
                df_1m, df_5m, df_4h = stream.get_dataframes('QQQ')
                if df_1m.empty:
                    print("⚠️  No data available yet...")
                    await asyncio.sleep(check_interval)
                    continue
                
                # Create fresh indicator with latest data
                indicator = ORBIndicator(df_1m, df_5m, df_4h, config)
                
                # CONSISTENT TIMEZONE: All times use same Eastern timezone
                today = now.date()
                today_ts = pd.Timestamp(today).tz_localize('US/Eastern')
                
                # Calculate market times - all Eastern timezone-aware
                # Remember: candle timestamps = START time, so 9:49 candle closes at 9:50
                market_open = pd.Timestamp(datetime.combine(today, config.market_open), tz=eastern)
                orb_period_end = market_open + timedelta(minutes=config.orb_minutes)  # 9:50 AM
                entry_start_time = orb_period_end  # 9:50 AM (can calculate ORB and start entries)
                entry_window_end = market_open + timedelta(minutes=60)  # 10:30 AM
                
                # Current market time - consistent timezone
                current_ts = pd.Timestamp(now)
                
                # Wait for ORB period to complete (until 9:50 when we can calculate ORB and start entries)
                if current_ts < entry_start_time:
                    minutes_until_entry = (entry_start_time - current_ts).total_seconds() / 60
                    if signal_state['last_check'] != 'waiting_orb_completion':
                        print(f"\n⏳ Waiting for ORB completion and entry start... {minutes_until_entry:.1f} minutes remaining")
                        signal_state['last_check'] = 'waiting_orb_completion'
                    await asyncio.sleep(check_interval)
                    continue
                
                # Check if we're past the entry window
                if current_ts > entry_window_end:
                    if not signal_state['signal_triggered'] and signal_state['last_check'] != 'window_passed':
                        print(f"\n⏰ Entry window closed at {entry_window_end.strftime('%H:%M')}")
                        print(f"📊 No ORB breakout detected today within 60-minute window")
                        signal_state['last_check'] = 'window_passed'
                    await asyncio.sleep(60)  # Check less frequently after window
                    continue
                
                # We're in the entry window (9:52 AM - 10:30 AM)
                minutes_into_window = (current_ts - entry_start_time).total_seconds() / 60
                minutes_remaining = (entry_window_end - current_ts).total_seconds() / 60
                
                # Calculate ORB levels if not done yet (at 9:50 when 9:49 candle closed)
                if not signal_state['orb_calculated'] and current_ts >= entry_start_time:
                    orb_levels = indicator.calculate_orb(today_ts)
                    if not pd.isna(orb_levels['orb_high']):
                        signal_state['orb_calculated'] = True
                        signal_state['orb_levels'] = orb_levels
                        print(f"\n📊 ORB LEVELS CALCULATED at {current_ts.strftime('%H:%M:%S')}")
                        print(f"   ORB High: ${orb_levels['orb_high']:.2f}")
                        print(f"   ORB Low: ${orb_levels['orb_low']:.2f}")
                        print(f"   ORB Range: ${orb_levels['orb_range']:.2f}")
                        print(f"\n🎯 Entry detection now active!")
                        print(f"   Entry window: {minutes_remaining:.1f} minutes remaining")
                        print("-" * 60)
                    else:
                        print("⚠️  Waiting for ORB calculation...")
                        await asyncio.sleep(check_interval)
                        continue
                
                # Display current price action (always show bars during entry window)
                latest_bars = indicator.df_1m
                if not latest_bars.empty:
                    # Get the most recent complete bar
                    latest_bar = latest_bars.iloc[-1]
                    latest_time = latest_bars.index[-1]
                    
                    # Only update display for new bars
                    if latest_time != last_bar_time:
                        last_bar_time = latest_time
                        
                        # Display current price action
                        status = "🟢 MONITORING" if not signal_state['signal_triggered'] else "✅ SIGNAL TRIGGERED"
                
                # Check for signal if not already triggered, ORB calculated, and past entry start time
                if (not signal_state['signal_triggered'] and 
                    signal_state['orb_calculated'] and 
                    current_ts >= entry_start_time):
                    if not latest_bars.empty:
                        # Check for breakout using scan logic with completed candle check
                        # Pass current time so it only checks completed candles
                        signal = indicator.scan_for_daily_signal(today_ts, up_to_time=current_ts)
                        
                        if signal:
                            signal_state['signal_triggered'] = True
                            signal_state['signal_details'] = signal
                            
                            # ALERT - SIGNAL DETECTED!
                            print(f"\n\n{'='*60}")
                            print(f"🚨🚨🚨 SIGNAL DETECTED! 🚨🚨🚨")
                            print(f"{'='*60}")
                            print(f"⏰ Time: {signal['time'].strftime('%H:%M:%S')}")
                            print(f"📈 Direction: {signal['type']}")
                            print(f"💰 Entry Price: ${signal['entry_price']:.2f}")
                            print(f"🎯 Confidence: {signal['confidence_level']} ({signal['confidence_score']:.1f})")
                            print(f"📏 ORB Range: ${signal['orb_range']:.2f}")
                            print(f"⚡ Breakout Strength: {signal['breakout_pct']:+.1f}%")
                            print(f"📊 Volume: {signal['volume']:,.0f}")
                            print(f"🕐 Entry Timing: {signal['minutes_after_open']:.0f} min after open")
                            
                            # Calculate trade levels
                            trade_levels = indicator.calculate_trade_levels(
                                signal['entry_price'],
                                signal['direction'],
                                signal['time']
                            )
                            
                            print(f"\n🎯 TRADE LEVELS:")
                            print(f"   Stop Loss: ${trade_levels['stop_loss']:.2f}")
                            print(f"   TP1 (50%): ${trade_levels['tp1']:.2f}")
                            print(f"   TP2 (25%): ${trade_levels['tp2']:.2f}")  
                            print(f"   TP3 (25%): ${trade_levels['tp3']:.2f}")
                            
                            if signal['confidence_level'] == 'HIGH':
                                print(f"\n🔥🔥🔥 HIGH CONFIDENCE SIGNAL 🔥🔥🔥")
                                print(f"STRONG SETUP - CONSIDER POSITION!")
                            
                            print(f"{'='*60}\n")
                            
                            # Save signal to file
                            signal_df = pd.DataFrame([signal])
                            filename = f"SIGNAL_{signal['type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                            signal_df.to_csv(filename, index=False)
                            print(f"💾 Signal saved to: {filename}")
                            
                            # Could add sound alert here
                            print("\a")  # Terminal bell
                            
                            # Continue monitoring for position management
                            print(f"\n📊 Signal triggered - monitoring position...")
                
                # Sleep before next check
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                print(f"\n⚠️  Error in monitoring loop: {e}")
                await asyncio.sleep(check_interval)
                
    except KeyboardInterrupt:
        print(f"\n\n⛔ Live monitoring stopped by user")
        
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print(f"\n🔌 Disconnecting from IBKR...")
        await stream.stop()
        print(f"✅ Disconnected")
        print(f"👋 Live monitor stopped")

if __name__ == "__main__":
    try:
        print("🚀 Starting ORB Live Signal Monitor...")
        print("📋 Requirements:")
        print("   • TWS running on localhost:7496")
        print("   • Market hours for signal detection")
        print("   • QQQ data subscription active")
        print("\n⚡ Press Ctrl+C to stop monitoring\n")
        
        asyncio.run(run_live_signal_monitor())
        
    except KeyboardInterrupt:
        print("\n👋 Live monitoring stopped")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)