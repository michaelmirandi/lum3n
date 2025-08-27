#!/usr/bin/env python3
"""
Live Data Streaming Module
Connects to IBKR, streams data, manages buffers with historical cache
"""
import asyncio
from datetime import datetime, timezone, timedelta
from ib_insync import IB, Stock
import pandas as pd
from typing import Dict, Optional

from .aggregator import CandleAggregator
from .cache import DataCache

# ANSI colors
CYAN  = "\033[36m"
RESET = "\033[0m"

class LiveDataStream:
    """
    Main data streaming class that:
    1. Loads historical data on startup
    2. Streams live 5-second bars from IBKR
    3. Aggregates into multiple timeframes
    4. Maintains DataFrames for indicator calculations
    """
    def __init__(self, config: dict, ib_client: IB = None):
        self.config = config
        self.ib = ib_client if ib_client is not None else IB()
        self.owns_ib_client = ib_client is None  # Track if we created the client
        self.aggregator = CandleAggregator(buffer_minutes=390)  # Full day buffer
        self.cache_manager = DataCache()
        self.contracts = {}
        self.symbols = config.get('symbols', ['QQQ'])
        
        # DataFrames - will be populated with historical + live data
        self.df_1m = {}
        self.df_5m = {}
        self.df_4h = {}
        
        # Connection settings
        self.host = config.get('host', '127.0.0.1')
        self.port = config.get('port', 7496)
        self.client_id = config.get('client_id', 101)
        
        self.connected = False
        self.streaming = False
        
    async def connect(self, retry_count: int = 5):
        """Connect to IBKR Gateway/TWS with automatic client ID retry"""
        original_client_id = self.client_id
        
        for attempt in range(retry_count):
            try:
                current_client_id = original_client_id + attempt
                print(f"🔌 Attempting connection with client ID {current_client_id}...")
                
                await self.ib.connectAsync(self.host, self.port, clientId=current_client_id)
                self.connected = True
                self.client_id = current_client_id  # Update to successful ID
                
                print(f"{CYAN}✅ Connected to IBKR at {self.host}:{self.port} (Client ID: {current_client_id}){RESET}")
                
                # Setup bar update handler
                self.ib.barUpdateEvent += self._on_bar_update
                
                return True
                
            except Exception as e:
                error_msg = str(e).lower()
                if "client id" in error_msg and "already in use" in error_msg:
                    print(f"⚠️ Client ID {current_client_id} in use, trying next...")
                    # Disconnect any partial connection
                    try:
                        if self.ib.isConnected():
                            self.ib.disconnect()
                    except:
                        pass
                    continue
                else:
                    print(f"❌ Connection failed (attempt {attempt + 1}): {e}")
                    if attempt == retry_count - 1:  # Last attempt
                        return False
                    await asyncio.sleep(1)  # Wait before retry
        
        print(f"❌ Failed to connect after {retry_count} attempts")
        return False
    
    async def initialize_data(self):
        """
        Load historical data and prepare for live streaming
        This ensures indicators have enough data at market open
        """
        print(f"\n{CYAN}═══ Initializing Historical Data ═══{RESET}")
        
        for symbol in self.symbols:
            print(f"\n📊 Processing {symbol}...")
            
            # Load historical data from cache or fetch from IBKR
            historical_data = await self.cache_manager.initialize_for_trading(
                self.ib, symbol
            )
            
            # Validate we have enough data
            if not self.cache_manager.validate_data(historical_data):
                print(f"⚠️ Warning: Insufficient historical data for {symbol}")
            
            # Store in our DataFrames
            self.df_1m[symbol] = historical_data.get('1m', pd.DataFrame())
            self.df_5m[symbol] = historical_data.get('5m', pd.DataFrame())
            self.df_4h[symbol] = historical_data.get('4h', pd.DataFrame())
            
            print(f"✅ {symbol} initialized with:")
            print(f"   1m: {len(self.df_1m[symbol])} bars")
            print(f"   5m: {len(self.df_5m[symbol])} bars")
            print(f"   4h: {len(self.df_4h[symbol])} bars")
        
        print(f"\n{CYAN}═══ Historical Data Ready ═══{RESET}\n")
    
    async def start_streaming(self):
        """Start streaming real-time bars"""
        if not self.connected:
            print("❌ Not connected to IBKR")
            return False
        
        print(f"\n{CYAN}═══ Starting Live Data Stream ═══{RESET}")
        
        for symbol in self.symbols:
            print(f"📡 Subscribing to {symbol}...")
            
            # Qualify contract
            contract = Stock(symbol, 'ARCA', 'USD')
            qualified = await self.ib.qualifyContractsAsync(contract)
            
            if not qualified:
                print(f"❌ Could not qualify {symbol}")
                continue
            
            self.contracts[symbol] = qualified[0]
            
            # Request real-time bars (5 seconds)
            self.ib.reqRealTimeBars(
                self.contracts[symbol],
                5,  # 5-second bars
                'TRADES',
                useRTH=False,  # Include pre/post market
                realTimeBarsOptions=[]
            )
            
            print(f"✅ Streaming {symbol}")
        
        self.streaming = True
        print(f"{CYAN}═══ Streaming Active ═══{RESET}\n")
        return True
    
    def _on_bar_update(self, bars, hasNewBar):
        """Handle incoming bar updates from IBKR"""
        if not hasNewBar:
            return
        
        symbol = bars.contract.symbol
        bar = bars[-1]
        
        # Convert to UTC timestamp
        ts_utc = bar.time.astimezone(timezone.utc)
        
        # Feed to aggregator
        self.aggregator.on_bar_5s(
            symbol,
            ts_utc,
            float(bar.open),
            float(bar.high),
            float(bar.low),
            float(bar.close),
            float(bar.volume or 0)
        )
        
        # Update our DataFrames with aggregated data
        self._update_dataframes(symbol)
    
    def _update_dataframes(self, symbol: str):
        """Update main DataFrames with latest aggregated data"""
        # Get latest DataFrames from aggregator
        agg_1m, agg_5m, agg_4h = self.aggregator.get_dataframes(symbol)
        
        # Merge with existing DataFrames (historical + new live data)
        if not agg_1m.empty:
            if symbol in self.df_1m and not self.df_1m[symbol].empty:
                # Find last timestamp in main DataFrame
                last_time = self.df_1m[symbol]['date'].max()
                # Append only new data
                new_data = agg_1m[agg_1m['date'] > last_time]
                if not new_data.empty:
                    self.df_1m[symbol] = pd.concat([
                        self.df_1m[symbol], new_data
                    ], ignore_index=True)
            else:
                self.df_1m[symbol] = agg_1m.copy()
        
        # Same for 5m
        if not agg_5m.empty:
            if symbol in self.df_5m and not self.df_5m[symbol].empty:
                last_time = self.df_5m[symbol]['date'].max()
                new_data = agg_5m[agg_5m['date'] > last_time]
                if not new_data.empty:
                    self.df_5m[symbol] = pd.concat([
                        self.df_5m[symbol], new_data
                    ], ignore_index=True)
            else:
                self.df_5m[symbol] = agg_5m.copy()
        
        # Same for 4h
        if not agg_4h.empty:
            if symbol in self.df_4h and not self.df_4h[symbol].empty:
                last_time = self.df_4h[symbol]['date'].max()
                new_data = agg_4h[agg_4h['date'] > last_time]
                if not new_data.empty:
                    self.df_4h[symbol] = pd.concat([
                        self.df_4h[symbol], new_data
                    ], ignore_index=True)
            else:
                self.df_4h[symbol] = agg_4h.copy()
    
    def get_dataframes(self, symbol: str) -> tuple:
        """Get current DataFrames for a symbol"""
        return (
            self.df_1m.get(symbol, pd.DataFrame()),
            self.df_5m.get(symbol, pd.DataFrame()),
            self.df_4h.get(symbol, pd.DataFrame())
        )
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol"""
        return self.aggregator.last_price.get(symbol)
    
    async def stop(self):
        """Stop streaming and disconnect"""
        print(f"\n{CYAN}═══ Stopping Data Stream ═══{RESET}")
        
        # Cancel real-time bars
        for symbol, contract in self.contracts.items():
            try:
                self.ib.cancelRealTimeBars(contract)
                print(f"✅ Stopped streaming {symbol}")
            except Exception as e:
                print(f"⚠️ Error stopping {symbol}: {e}")
        
        # Flush aggregator
        try:
            self.aggregator.flush()
        except Exception as e:
            print(f"⚠️ Error flushing aggregator: {e}")
        
        # Disconnect only if we own the client
        if (self.connected or self.ib.isConnected()) and self.owns_ib_client:
            try:
                # Give it a moment to process cancellations
                await asyncio.sleep(0.5)
                self.ib.disconnect()
                self.connected = False
                print(f"✅ Disconnected from IBKR (Client ID {self.client_id})")
            except Exception as e:
                print(f"⚠️ Error during disconnect: {e}")
                # Force disconnect
                try:
                    self.ib.disconnect()
                except:
                    pass
                self.connected = False
        elif not self.owns_ib_client:
            print(f"✅ Released shared IBKR client (Client ID {self.client_id})")
            self.connected = False
        
        self.streaming = False
        print(f"{CYAN}═══ Shutdown Complete ═══{RESET}")
    
    async def run_test(self, duration_seconds: int = 300):
        """Run a test stream for specified duration"""
        print(f"\n🧪 Running {duration_seconds} second test...")
        
        # Connect
        if not await self.connect():
            return
        
        # Initialize historical data
        await self.initialize_data()
        
        # Start streaming
        if not await self.start_streaming():
            return
        
        # Run for specified duration
        await asyncio.sleep(duration_seconds)
        
        # Show summary
        print(f"\n{CYAN}═══ Test Summary ═══{RESET}")
        for symbol in self.symbols:
            df_1m = self.df_1m.get(symbol, pd.DataFrame())
            df_5m = self.df_5m.get(symbol, pd.DataFrame())
            df_4h = self.df_4h.get(symbol, pd.DataFrame())
            
            print(f"\n{symbol}:")
            print(f"  1m bars: {len(df_1m)}")
            print(f"  5m bars: {len(df_5m)}")
            print(f"  4h bars: {len(df_4h)}")
            
            if not df_1m.empty:
                print(f"  Latest 1m: {df_1m.iloc[-1]['date']} - ${df_1m.iloc[-1]['close']:.2f}")
        
        # Stop
        await self.stop()