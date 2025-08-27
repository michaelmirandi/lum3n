#!/usr/bin/env python3
"""
Core ORB System - Direct port from orb_trading_system.py
Contains ORBConfig, Trade, and ORBIndicator classes with all proven logic
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime, time, timedelta
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

class ExitReason(Enum):
    TP1 = "TP1"
    TP2 = "TP2" 
    TP3 = "TP3"
    STOP_LOSS = "STOP_LOSS"
    TIME_LIMIT = "TIME_LIMIT"

class ConfidenceLevel(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

@dataclass
class ORBConfig:
    """Configuration parameters for ORB trading system with 4H confluence"""
    orb_minutes: int = 20
    use_volume_filter: bool = True
    atr_multiplier: float = 2.5
    max_hold_minutes: int = 120
    risk_per_trade: float = 1000.0
    one_trade_per_day: bool = True
    use_confidence_filter: bool = False
    min_confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    rsi_period: int = 14
    ema_period: int = 9
    volume_ma_period: int = 20
    initial_capital: float = 100000.0
    market_open: time = time(9, 30)
    market_close: time = time(16, 0)
    
    # Dynamic stop loss parameters
    initial_atr_multiplier: float = 1.5
    final_atr_multiplier: float = 0.5
    narrowing_start_minutes: int = 45
    stop_update_frequency_minutes: int = 5
    
    # 4H Multi-Timeframe Confluence Parameters
    use_4h_confluence: bool = True
    
    # 4H Indicator Settings
    rsi_4h_period: int = 14
    macd_4h_fast: int = 12
    macd_4h_slow: int = 26
    macd_4h_signal: int = 9
    
    # 4H RSI Confluence Thresholds
    rsi_4h_bullish_min: float = 40.0
    rsi_4h_bullish_max: float = 70.0
    rsi_4h_bearish_min: float = 30.0
    rsi_4h_bearish_max: float = 60.0
    
    # 4H MACD Requirements
    require_macd_4h_alignment: bool = True
    
    # Enhanced Confidence Scoring Weights
    confidence_4h_rsi_weight: float = 2.0
    confidence_4h_macd_weight: float = 1.5
    confidence_4h_max_total: float = 5.5
    
    # Enhanced Confidence Level Thresholds
    confidence_high_threshold: float = 6.0
    confidence_medium_threshold: float = 4.0
    confidence_low_threshold: float = 0.0
    
    # Entry Window Parameters (from notebook)
    entry_window_minutes: int = 60  # 60-minute window after market open for entries
    
    @classmethod
    def from_yaml(cls, config_path: str = None):
        """Load configuration from YAML file"""
        if config_path is None:
            # Default to config.yaml in the project root
            config_path = Path(__file__).parent.parent.parent / "config.yaml"
        
        if not Path(config_path).exists():
            print(f"⚠️ Config file not found at {config_path}, using defaults")
            return cls()
        
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Convert time strings to time objects
            if 'market_open' in config_data:
                market_open_str = config_data['market_open']
                hour, minute = map(int, market_open_str.split(':'))
                config_data['market_open'] = time(hour, minute)
            
            if 'market_close' in config_data:
                market_close_str = config_data['market_close']
                hour, minute = map(int, market_close_str.split(':'))
                config_data['market_close'] = time(hour, minute)
            
            # Convert confidence level string to enum
            if 'min_confidence_level' in config_data:
                level_str = config_data['min_confidence_level']
                config_data['min_confidence_level'] = ConfidenceLevel(level_str)
            
            # Create instance with config data
            return cls(**{k: v for k, v in config_data.items() if hasattr(cls, k)})
            
        except Exception as e:
            print(f"⚠️ Error loading config from {config_path}: {e}")
            print("Using default configuration")
            return cls()

@dataclass
class Trade:
    """Trade record for tracking individual trades"""
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp] = None
    direction: Optional[TradeDirection] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    shares: int = 0
    stop_loss: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0
    exit_reason: Optional[ExitReason] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    confidence_score: float = 0.0
    confidence_level: Optional[ConfidenceLevel] = None
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    holding_minutes: int = 0
    orb_high: float = 0.0
    orb_low: float = 0.0
    # Dynamic stop loss tracking
    initial_stop_loss: float = 0.0
    atr_value: float = 0.0
    stop_history: List[Tuple[pd.Timestamp, float]] = field(default_factory=list)
    narrowing_start_time: Optional[pd.Timestamp] = None

class ORBIndicator:
    """Opening Range Breakout indicator with all technical analysis components"""
    
    def __init__(self, df_1m: pd.DataFrame, df_5m: pd.DataFrame, df_4h: pd.DataFrame, config: ORBConfig):
        self.df_1m = df_1m.copy()
        self.df_5m = df_5m.copy()
        self.df_4h = df_4h.copy()
        self.config = config
        
        # Prepare data for all timeframes
        self._prepare_data()
        
        # Calculate all technical indicators including 4H
        self._calculate_indicators()
        
        # Daily ORB levels cache
        self.orb_levels = {}
        
    def _prepare_data(self):
        """Prepare and clean the data for all timeframes"""
        # Convert date columns to datetime if they're strings
        for df in [self.df_1m, self.df_5m, self.df_4h]:
            if df['date'].dtype == 'object':
                df['date'] = pd.to_datetime(df['date'])
            
            # Handle timezone-aware data - convert to naive UTC for consistency
            if df['date'].dt.tz is not None:
                df['date'] = df['date'].dt.tz_convert('US/Eastern').dt.tz_localize(None)
            
            # Set date as index for easier operations
            df.set_index('date', inplace=True)
            
            # Add time-based columns
            df['time'] = df.index.time
            df['date_only'] = df.index.date
            
        # Sort by timestamp to ensure chronological order
        self.df_1m.sort_index(inplace=True)
        self.df_5m.sort_index(inplace=True)
        self.df_4h.sort_index(inplace=True)
        
    def _calculate_indicators(self):
        """Calculate all technical indicators for all timeframes"""
        print("🔧 Calculating technical indicators...")
        
        # Calculate indicators for 1-minute data
        self._calculate_rsi(self.df_1m, period=self.config.rsi_period)
        self._calculate_ema(self.df_1m, period=self.config.ema_period)
        self._calculate_macd(self.df_1m)
        self._calculate_volume_ma(self.df_1m, period=self.config.volume_ma_period)
        self._calculate_vwap(self.df_1m)
        
        # Calculate indicators for 5-minute data - MATCH BACKTESTING EXACTLY
        self._calculate_atr(self.df_5m, period=14)
        self._calculate_ema(self.df_5m, period=21, column_name='ema_21')
        # ❌ REMOVED to match backtesting: RSI, MACD, Volume MA, VWAP
        # This forces fallback to 1m scoring like in backtesting
        
        # Calculate indicators for 4-hour data (for confluence)
        if self.config.use_4h_confluence:
            self._calculate_4h_indicators()
        
        print("✅ All indicators calculated successfully")
        
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14):
        """Calculate RSI indicator"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
    def _calculate_ema(self, df: pd.DataFrame, period: int = 9, column_name: str = None):
        """Calculate Exponential Moving Average"""
        if column_name is None:
            column_name = f'ema_{period}'
        df[column_name] = df['close'].ewm(span=period, adjust=False).mean()
        
    def _calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD indicator"""
        ema_fast = df['close'].ewm(span=fast).mean()
        ema_slow = df['close'].ewm(span=slow).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = true_range.rolling(window=period).mean()
        
    def _calculate_volume_ma(self, df: pd.DataFrame, period: int = 20):
        """Calculate Volume Moving Average"""
        df['volume_ma'] = df['volume'].rolling(window=period).mean()
        
    def _calculate_vwap(self, df: pd.DataFrame):
        """Calculate Volume Weighted Average Price (daily reset)"""
        df['vwap'] = (df.groupby('date_only').apply(
            lambda x: (x['close'] * x['volume']).cumsum() / x['volume'].cumsum()
        )).reset_index(level=0, drop=True)
    
    def _calculate_4h_indicators(self):
        """Calculate 4H timeframe indicators for multi-timeframe confluence"""
        print("🔧 Calculating 4H confluence indicators...")
        
        # Calculate 4H RSI
        self._calculate_rsi(self.df_4h, period=self.config.rsi_4h_period)
        
        # Calculate 4H MACD
        self._calculate_macd(self.df_4h, 
                           fast=self.config.macd_4h_fast,
                           slow=self.config.macd_4h_slow, 
                           signal=self.config.macd_4h_signal)
        
        # Calculate 4H Moving Average for distance analysis
        self._calculate_ema(self.df_4h, period=20, column_name='ema_20')
        
        print(f"✅ 4H indicators calculated for {len(self.df_4h)} bars")
    
    def calculate_orb(self, date: pd.Timestamp) -> Dict[str, float]:
        """Calculate Opening Range for a specific date"""
        date_str = date.date()
        
        # Check cache first
        if date_str in self.orb_levels:
            return self.orb_levels[date_str]
        
        # Filter 5-minute data for the specific date
        day_data = self.df_5m[self.df_5m['date_only'] == date_str].copy()
        
        if day_data.empty:
            return {'orb_high': np.nan, 'orb_low': np.nan, 'orb_range': 0}
        
        # Find market open time - ensure it's timezone naive
        market_open_time = pd.Timestamp(datetime.combine(date_str, self.config.market_open))
        orb_end_time = market_open_time + timedelta(minutes=self.config.orb_minutes)
        
        # Get opening range bars (first X 5-minute candles based on config)
        orb_data = day_data[
            (day_data.index >= market_open_time) & 
            (day_data.index < orb_end_time)
        ]
        
        if len(orb_data) < 3:  # Need at least 3 bars for valid ORB
            return {'orb_high': np.nan, 'orb_low': np.nan, 'orb_range': 0}
        
        orb_high = orb_data['high'].max()
        orb_low = orb_data['low'].min()
        orb_range = orb_high - orb_low
        
        result = {
            'orb_high': orb_high,
            'orb_low': orb_low,
            'orb_range': orb_range
        }
        
        # Cache the result
        self.orb_levels[date_str] = result
        
        return result
    
    def get_5m_indicators_at_time(self, timestamp: pd.Timestamp) -> Optional[Dict[str, float]]:
        """Get 5M indicators for the most recent 5M bar at/before timestamp - ENHANCED VERSION"""
        try:
            # Find the most recent 5M bar at or before the given timestamp
            available_5m_times = self.df_5m.index[self.df_5m.index <= timestamp]
            
            if len(available_5m_times) == 0:
                return None
            
            nearest_5m_time = available_5m_times[-1]
            bar_5m = self.df_5m.loc[nearest_5m_time]
            
            # Check if we have valid indicator data
            if pd.isna(bar_5m.get('rsi', np.nan)) or pd.isna(bar_5m.get('macd', np.nan)):
                return None
            
            age_minutes = (timestamp - nearest_5m_time).total_seconds() / 60
            
            return {
                'rsi_5m': float(bar_5m['rsi']),
                'macd_5m': float(bar_5m['macd']),
                'macd_signal_5m': float(bar_5m['macd_signal']),
                'macd_histogram_5m': float(bar_5m['macd_histogram']),
                'volume_5m': float(bar_5m['volume']),
                'volume_ma_5m': float(bar_5m['volume_ma']),
                'ema_21_5m': float(bar_5m['ema_21']),
                'timestamp_5m': nearest_5m_time,
                'age_minutes': age_minutes,
                'close_5m': float(bar_5m['close']),
                'vwap_5m': float(bar_5m.get('vwap', bar_5m['close']))  # Use close if VWAP not available
            }
            
        except Exception:
            return None
    
    def get_4h_indicators_at_time(self, timestamp: pd.Timestamp) -> Optional[Dict[str, float]]:
        """Get 4H RSI and MACD values for the most recent 4H bar at/before timestamp"""
        try:
            # Find the most recent 4H bar at or before the given timestamp
            available_4h_times = self.df_4h.index[self.df_4h.index <= timestamp]
            
            if len(available_4h_times) == 0:
                return None
            
            nearest_4h_time = available_4h_times[-1]
            bar_4h = self.df_4h.loc[nearest_4h_time]
            
            # Check if we have valid indicator data
            if pd.isna(bar_4h.get('rsi', np.nan)) or pd.isna(bar_4h.get('macd', np.nan)):
                return None
            
            age_minutes = (timestamp - nearest_4h_time).total_seconds() / 60
            
            return {
                'rsi_4h': float(bar_4h['rsi']),
                'macd_4h': float(bar_4h['macd']),
                'macd_signal_4h': float(bar_4h['macd_signal']),
                'macd_histogram_4h': float(bar_4h['macd_histogram']),
                'ema_20_4h': float(bar_4h.get('ema_20', np.nan)),
                'timestamp_4h': nearest_4h_time,
                'age_minutes': age_minutes,
                'close_4h': float(bar_4h['close'])
            }
            
        except Exception as e:
            print(f"Warning: Error getting 4H indicators at {timestamp}: {e}")
            return None
    
    def check_entry_signal(self, timestamp: pd.Timestamp, direction: TradeDirection) -> bool:
        """Check if entry conditions are met using notebook proven logic"""
        try:
            # Get current 1-minute data (entry price based on 1m close)
            current_1m = self.df_1m.loc[timestamp] if timestamp in self.df_1m.index else None
            
            if current_1m is None:
                return False
            
            # Get ORB levels for the day
            orb_levels = self.calculate_orb(timestamp)
            if np.isnan(orb_levels['orb_high']):
                return False
            
            # Check if we're in the valid entry window
            market_open = pd.Timestamp(datetime.combine(timestamp.date(), self.config.market_open))
            orb_end = market_open + timedelta(minutes=self.config.orb_minutes)  # 9:50 AM
            entry_window_end = market_open + timedelta(minutes=60)  # 10:30 AM
            
            # Must be after ORB completion but within 60 minutes of market open
            if timestamp < orb_end or timestamp > entry_window_end:
                return False
            
            # Exact notebook logic: HIGH/LOW breaks ORB AND CLOSE confirms breakout
            if direction == TradeDirection.LONG:
                # For LONG: 1m high breaks orb_high AND 1m close is above orb_high
                high_breaks_orb = current_1m['high'] > orb_levels['orb_high']
                close_confirms = current_1m['close'] > orb_levels['orb_high']
                return high_breaks_orb and close_confirms
            else:
                # For SHORT: 1m low breaks orb_low AND 1m close is below orb_low  
                low_breaks_orb = current_1m['low'] < orb_levels['orb_low']
                close_confirms = current_1m['close'] < orb_levels['orb_low']
                return low_breaks_orb and close_confirms
            
        except (KeyError, IndexError) as e:
            print(f"Warning: Entry signal check failed: {e}")
            return False
    
    def calculate_confidence_score(self, timestamp: pd.Timestamp, direction: TradeDirection) -> Tuple[float, ConfidenceLevel]:
        """Enhanced confidence score with 5M base + 4H confluence for cleaner signals"""
        try:
            # Calculate base 5-minute confidence score
            base_score = self._calculate_5m_confidence_score(timestamp, direction)
            
            # Add 4H confluence if enabled
            confluence_score = 0.0
            if self.config.use_4h_confluence:
                confluence_score = self._calculate_4h_confluence_score(timestamp, direction)
            
            total_score = base_score + confluence_score
            
            # Determine confidence level
            if total_score >= self.config.confidence_high_threshold:
                confidence_level = ConfidenceLevel.HIGH
            elif total_score >= self.config.confidence_medium_threshold:
                confidence_level = ConfidenceLevel.MEDIUM
            else:
                confidence_level = ConfidenceLevel.LOW
                
            return total_score, confidence_level
            
        except Exception as e:
            print(f"Warning: Confidence score calculation failed: {e}")
            return 0.0, ConfidenceLevel.LOW
    
    def _calculate_5m_confidence_score(self, timestamp: pd.Timestamp, direction: TradeDirection) -> float:
        """Calculate BALANCED 5-minute confidence score - QUALITY OVER QUANTITY"""
        indicators_5m = self.get_5m_indicators_at_time(timestamp)
        if not indicators_5m:
            # Fallback to 1m data if 5m not available
            try:
                current_1m = self.df_1m.loc[timestamp]
                return self._calculate_1m_fallback_score(current_1m, direction)
            except:
                return 0.5  # Low score if no data - don't reward blindly
            
        score = 0.0
        rsi = indicators_5m['rsi_5m']
        
        if direction == TradeDirection.LONG:
            # RSI scoring for LONG
            if 50 <= rsi <= 65:
                score += 3.0
            elif 45 <= rsi < 50:
                score += 2.0
            elif 65 < rsi <= 75:
                score += 1.0
            elif 35 <= rsi < 45:
                score += 1.5
                
            # MACD scoring
            macd_hist = indicators_5m['macd_histogram_5m']
            if macd_hist > 0.05:
                score += 1.5
            elif macd_hist > 0:
                score += 0.5
                
            if indicators_5m['macd_5m'] > indicators_5m['macd_signal_5m']:
                cross_strength = indicators_5m['macd_5m'] - indicators_5m['macd_signal_5m']
                if cross_strength > 0.02:
                    score += 1.0
                elif cross_strength > 0:
                    score += 0.5
                
            # Volume scoring
            volume_ratio = indicators_5m['volume_5m'] / indicators_5m['volume_ma_5m']
            if volume_ratio > 1.5:
                score += 1.0
            elif volume_ratio > 1.1:
                score += 0.5
                
            # VWAP scoring
            if indicators_5m['close_5m'] > indicators_5m['vwap_5m']:
                price_distance = (indicators_5m['close_5m'] - indicators_5m['vwap_5m']) / indicators_5m['vwap_5m'] * 100
                if price_distance > 0.1:
                    score += 1.0
                else:
                    score += 0.3
                
        else:  # SHORT trades
            # RSI scoring for SHORT
            if 35 <= rsi <= 50:
                score += 3.0
            elif 50 < rsi <= 55:
                score += 2.0
            elif 25 <= rsi < 35:
                score += 1.0
            elif 55 < rsi <= 65:
                score += 1.5
                
            # MACD scoring (bearish)
            macd_hist = indicators_5m['macd_histogram_5m']
            if macd_hist < -0.05:
                score += 1.5
            elif macd_hist < 0:
                score += 0.5
                
            if indicators_5m['macd_5m'] < indicators_5m['macd_signal_5m']:
                cross_strength = indicators_5m['macd_signal_5m'] - indicators_5m['macd_5m']
                if cross_strength > 0.02:
                    score += 1.0
                elif cross_strength > 0:
                    score += 0.5
                
            # Volume scoring
            volume_ratio = indicators_5m['volume_5m'] / indicators_5m['volume_ma_5m']
            if volume_ratio > 1.5:
                score += 1.0
            elif volume_ratio > 1.1:
                score += 0.5
                
            # VWAP scoring (bearish)
            if indicators_5m['close_5m'] < indicators_5m['vwap_5m']:
                price_distance = (indicators_5m['vwap_5m'] - indicators_5m['close_5m']) / indicators_5m['vwap_5m'] * 100
                if price_distance > 0.1:
                    score += 1.0
                else:
                    score += 0.3
        
        return max(0.0, min(score, 7.0))  # Cap at 7.0
    
    def _calculate_1m_fallback_score(self, current_1m, direction: TradeDirection) -> float:
        """Selective fallback scoring using 1m data when 5m not available - EXACT COPY"""
        score = 0.0  # Start from zero - must earn points
        try:
            rsi = current_1m.get('rsi', 50)
            
            if direction == TradeDirection.LONG:
                # Be selective with 1m RSI (noisier)
                if 45 <= rsi <= 70:
                    score += 2.5
                elif 35 <= rsi < 45:
                    score += 1.0
                    
                if current_1m.get('macd_histogram', 0) > 0.01:  # Meaningful positive
                    score += 1.0
                if current_1m.get('close', 0) > current_1m.get('vwap', 0):
                    score += 0.5
            else:  # SHORT
                # Be selective with 1m RSI (noisier)
                if 30 <= rsi <= 55:
                    score += 2.5
                elif 55 < rsi <= 65:
                    score += 1.0
                    
                if current_1m.get('macd_histogram', 0) < -0.01:  # Meaningful negative
                    score += 1.0
                if current_1m.get('close', 0) < current_1m.get('vwap', 0):
                    score += 0.5
        except:
            return 1.0  # Minimal fallback score
            
        return min(score, 5.0)  # Lower cap for 1m fallback
    
    def _calculate_4h_confluence_score(self, timestamp: pd.Timestamp, direction: TradeDirection) -> float:
        """Calculate 4H timeframe confluence score - EXACT COPY from orb_trading_system.py (max 5.5 points)"""
        
        indicators_4h = self.get_4h_indicators_at_time(timestamp)
        if not indicators_4h:
            return 0.0
        
        score = 0.0
        rsi = indicators_4h['rsi_4h']
        close_price = indicators_4h['close_4h']
        ema_20 = indicators_4h['ema_20_4h']
        
        # Calculate RSI slope (momentum direction over last 8-12 hours)
        rsi_slope = self._get_rsi_slope_4h(timestamp)
        
        # Calculate distance from moving average (% above/below)
        ma_distance_pct = ((close_price - ema_20) / ema_20) * 100
        
        if direction == TradeDirection.LONG:
            # 1. RSI ZONE SCORING (max 3.0 points, min -1.5 points) - MORE GENEROUS
            if 50 <= rsi <= 75:
                score += 3.0  # PERFECT ZONE - wider range, more generous scoring
            elif rsi > 80:
                score -= 1.5  # SOFT PENALTY - discourage but don't eliminate
            elif 75 < rsi <= 80:
                score -= 0.5  # LIGHT PENALTY - gentle discouragement
            # No penalty for RSI < 50 - just no bonus
            
            # 2. RSI SLOPE ANALYSIS (BONUS ONLY: 0 to 1.0 points)
            if rsi_slope > 0.5:
                score += 1.0  # RSI trending up strongly - momentum building
            elif rsi_slope > 0:
                score += 0.5  # RSI trending up - good momentum
            # No penalty for negative slope - just no bonus
                
            # 3. MOVING AVERAGE DISTANCE (BONUS ONLY: 0 to 0.5 points)
            if -1.0 <= ma_distance_pct <= 2.0:
                score += 0.5  # Sweet spot near MA - room to run without extension
            # No penalty for being far from MA - just no bonus
            
            # 4. MACD CONFIRMATION (max 1.0 points)
            if indicators_4h['macd_histogram_4h'] > 0:
                score += 0.5  # MACD histogram positive
            if indicators_4h['macd_4h'] > indicators_4h['macd_signal_4h']:
                score += 0.5  # MACD above signal line
                
        else:  # SHORT direction
            # 1. RSI ZONE SCORING (max 3.0 points, min -1.5 points) - MORE GENEROUS
            if 25 <= rsi <= 50:
                score += 3.0  # PERFECT ZONE - wider range, more generous scoring
            elif rsi < 20:
                score -= 1.5  # SOFT PENALTY - discourage but don't eliminate
            elif 20 <= rsi < 25:
                score -= 0.5  # LIGHT PENALTY - gentle discouragement
            # No penalty for RSI > 50 - just no bonus
            
            # 2. RSI SLOPE ANALYSIS (BONUS ONLY: 0 to 1.0 points)
            if rsi_slope < -0.5:
                score += 1.0  # RSI trending down strongly - momentum building
            elif rsi_slope < 0:
                score += 0.5  # RSI trending down - good momentum
            # No penalty for positive slope - just no bonus
                
            # 3. MOVING AVERAGE DISTANCE (BONUS ONLY: 0 to 0.5 points)
            if -2.0 <= ma_distance_pct <= 1.0:
                score += 0.5  # Sweet spot near MA - room to fall without overextension
            # No penalty for being far from MA - just no bonus
            
            # 4. MACD CONFIRMATION (max 1.0 points)
            if indicators_4h['macd_histogram_4h'] < 0:
                score += 0.5  # MACD histogram negative
            if indicators_4h['macd_4h'] < indicators_4h['macd_signal_4h']:
                score += 0.5  # MACD below signal line
        
        # Cap the confluence score at the configured maximum
        return min(score, self.config.confidence_4h_max_total)
    
    def scan_for_entry_signal(self, date: pd.Timestamp) -> Optional[Dict[str, Union[pd.Timestamp, TradeDirection, float, Dict]]]:
        """
        Scan the entire entry window (9:50 AM - 10:30 AM) for the first valid breakout signal
        Returns the first signal found or None if no signals
        """
        try:
            market_open = pd.Timestamp(datetime.combine(date.date(), self.config.market_open))
            orb_end = market_open + timedelta(minutes=self.config.orb_minutes)  # 9:50 AM
            entry_window_end = market_open + timedelta(minutes=60)  # 10:30 AM
            
            # Get all 5m bars in the entry window
            entry_window_bars = self.df_5m.index[
                (self.df_5m.index >= orb_end) & 
                (self.df_5m.index <= entry_window_end)
            ]
            
            if len(entry_window_bars) == 0:
                return None
            
            # Scan chronologically for first valid signal
            for timestamp in entry_window_bars:
                # Check LONG first, then SHORT
                for direction in [TradeDirection.LONG, TradeDirection.SHORT]:
                    if self.check_entry_signal(timestamp, direction):
                        # Found a signal! Get all the details
                        confidence_score, confidence_level = self.calculate_confidence_score(timestamp, direction)
                        
                        # Get entry price from 5m close
                        entry_price = float(self.df_5m.loc[timestamp]['close'])
                        
                        # Calculate trade levels
                        trade_levels = self.calculate_trade_levels(entry_price, direction, timestamp)
                        
                        # Get ORB levels
                        orb_levels = self.calculate_orb(timestamp)
                        
                        return {
                            'timestamp': timestamp,
                            'direction': direction,
                            'entry_price': entry_price,
                            'confidence_score': confidence_score,
                            'confidence_level': confidence_level,
                            'trade_levels': trade_levels,
                            'orb_levels': orb_levels,
                            'minutes_after_open': (timestamp - market_open).total_seconds() / 60
                        }
            
            # No signals found in the entire window
            return None
            
        except Exception as e:
            print(f"Warning: Error scanning for entry signal on {date}: {e}")
            return None
    
    def calculate_trade_levels(self, entry_price: float, direction: TradeDirection, 
                             timestamp: pd.Timestamp) -> Dict[str, float]:
        """Calculate trade levels (stop loss, take profits)"""
        orb_levels = self.calculate_orb(timestamp)
        orb_range = orb_levels['orb_range']
        
        # Get ATR from 5-minute data
        atr_value = self.df_5m.loc[timestamp]['atr'] if timestamp in self.df_5m.index else orb_range * 0.5
        
        if direction == TradeDirection.LONG:
            # Initial wide stop loss
            initial_stop = entry_price - (atr_value * self.config.initial_atr_multiplier)
            
            # Take profit levels using Fibonacci extensions
            tp1 = entry_price + (orb_range * 1.272)  # 127.2% extension
            tp2 = entry_price + (orb_range * 1.618)  # 161.8% extension
            tp3 = entry_price + (orb_range * 2.0)    # 200% extension
            
        else:  # SHORT
            # Initial wide stop loss
            initial_stop = entry_price + (atr_value * self.config.initial_atr_multiplier)
            
            # Take profit levels using Fibonacci extensions
            tp1 = entry_price - (orb_range * 1.272)
            tp2 = entry_price - (orb_range * 1.618)
            tp3 = entry_price - (orb_range * 2.0)
        
        return {
            'stop_loss': initial_stop,
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'atr_value': atr_value,
            'orb_range': orb_range
        }
    
    def _get_rsi_slope_4h(self, timestamp: pd.Timestamp) -> float:
        """Calculate RSI slope over last 2-3 bars (8-12 hours) for momentum analysis"""
        try:
            # Find recent 4H bars
            available_times = self.df_4h.index[self.df_4h.index <= timestamp]
            if len(available_times) < 3:
                return 0.0  # Not enough data
            
            # Get last 3 bars
            recent_times = available_times[-3:]
            recent_rsi = self.df_4h.loc[recent_times, 'rsi'].values
            
            # Check for valid RSI values
            if any(pd.isna(recent_rsi)):
                return 0.0
            
            # Calculate slope: (newest - oldest) / periods
            rsi_slope = (recent_rsi[-1] - recent_rsi[0]) / 2.0
            
            return rsi_slope
            
        except Exception:
            return 0.0
    
    def get_nearest_5m_atr(self, timestamp: pd.Timestamp) -> float:
        """Get the nearest 5-minute ATR value for a given timestamp"""
        try:
            # Find the nearest 5-minute bar at or before the given timestamp
            available_times = self.df_5m.index[self.df_5m.index <= timestamp]
            if len(available_times) == 0:
                return 2.0  # Default ATR if none available
                
            nearest_time = available_times[-1]
            return self.df_5m.loc[nearest_time, 'atr']
        except:
            return 2.0  # Default ATR
    
    def get_nearest_5m_ema21(self, timestamp: pd.Timestamp) -> float:
        """Get the nearest 5-minute EMA21 value for a given timestamp"""
        try:
            # Find the nearest 5-minute bar at or before the given timestamp
            available_times = self.df_5m.index[self.df_5m.index <= timestamp]
            if len(available_times) == 0:
                # Get current price as fallback
                current_price = self.df_1m.loc[timestamp, 'close'] if timestamp in self.df_1m.index else 570.0
                return current_price
                
            nearest_time = available_times[-1]
            return self.df_5m.loc[nearest_time, 'ema_21']
        except:
            # Get current price as fallback
            try:
                return self.df_1m.loc[timestamp, 'close'] if timestamp in self.df_1m.index else 570.0
            except:
                return 570.0  # Default fallback price