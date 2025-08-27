#!/usr/bin/env python3
"""
Opening Range Breakout (ORB) Trading System Implementation
Complete system with backtesting capabilities for equity trading
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    orb_minutes: int = 15
    use_volume_filter: bool = True
    atr_multiplier: float = 1.5
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
    initial_atr_multiplier: float = 3.0
    final_atr_multiplier: float = 1.5
    narrowing_start_minutes: int = 45
    stop_update_frequency_minutes: int = 5
    
    # 4H Multi-Timeframe Confluence Parameters
    use_4h_confluence: bool = True
    
    # 4H Indicator Settings
    rsi_4h_period: int = 14
    macd_4h_fast: int = 12
    macd_4h_slow: int = 26
    macd_4h_signal: int = 9
    
    # 4H RSI Confluence Thresholds - OPTIONS OPTIMIZED
    rsi_4h_bullish_min: float = 40.0      # Minimum for bullish bias
    rsi_4h_bullish_max: float = 70.0      # Maximum before overbought
    rsi_4h_bearish_min: float = 30.0      # Minimum before oversold  
    rsi_4h_bearish_max: float = 60.0      # Maximum for bearish bias
    
    # 4H MACD Requirements
    require_macd_4h_alignment: bool = True
    
    # Enhanced Confidence Scoring Weights
    confidence_4h_rsi_weight: float = 2.0      # Max points from 4H RSI
    confidence_4h_macd_weight: float = 1.5     # Max points from 4H MACD
    confidence_4h_max_total: float = 5.5       # Cap total 4H contribution (sophisticated system)
    
    # Enhanced Confidence Level Thresholds (5M base + 4H confluence: ~13 max total)
    confidence_high_threshold: float = 6     # High confidence for options trading (5M + 4H)
    confidence_medium_threshold: float = 4.0   # Medium confidence threshold (cleaner signals)
    confidence_low_threshold: float = 0.0      # Minimum threshold

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
        # Calculate indicators for 1-minute data
        self._calculate_rsi(self.df_1m, period=self.config.rsi_period)
        self._calculate_ema(self.df_1m, period=self.config.ema_period)
        self._calculate_macd(self.df_1m)
        self._calculate_volume_ma(self.df_1m, period=self.config.volume_ma_period)
        self._calculate_vwap(self.df_1m)
        
        # Calculate indicators for 5-minute data
        self._calculate_atr(self.df_5m, period=14)
        self._calculate_ema(self.df_5m, period=21, column_name='ema_21')
        
        # Calculate indicators for 4-hour data (for confluence)
        if self.config.use_4h_confluence:
            self._calculate_4h_indicators()
        
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
        print(f"   4H RSI range: {self.df_4h['rsi'].min():.1f} - {self.df_4h['rsi'].max():.1f}")
        print(f"   4H MACD valid bars: {self.df_4h['macd'].notna().sum()}")
        print(f"   4H EMA-20 valid bars: {self.df_4h['ema_20'].notna().sum()}")
    
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
            if pd.isna(bar_4h.get('rsi', np.nan)) or pd.isna(bar_4h.get('macd', np.nan)) or pd.isna(bar_4h.get('ema_20', np.nan)):
                return None
            
            age_minutes = (timestamp - nearest_4h_time).total_seconds() / 60
            
            return {
                'rsi_4h': float(bar_4h['rsi']),
                'macd_4h': float(bar_4h['macd']),
                'macd_signal_4h': float(bar_4h['macd_signal']),
                'macd_histogram_4h': float(bar_4h['macd_histogram']),
                'ema_20_4h': float(bar_4h['ema_20']),
                'timestamp_4h': nearest_4h_time,
                'age_minutes': age_minutes,
                'close_4h': float(bar_4h['close'])
            }
            
        except Exception as e:
            print(f"Warning: Error getting 4H indicators at {timestamp}: {e}")
            return None
        
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
        
        # Get opening range bars (first 3 5-minute candles = 15 minutes)
        orb_data = day_data[
            (day_data.index >= market_open_time) & 
            (day_data.index < orb_end_time)
        ]
        
        if len(orb_data) < 3:  # Need at least 3 bars for 15-minute ORB
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
        
    def check_entry_signal(self, timestamp: pd.Timestamp, direction: TradeDirection) -> bool:
        """Check if entry conditions are met for the given direction"""
        try:
            # Get current 5-minute and 1-minute data
            current_5m = self.df_5m.loc[timestamp] if timestamp in self.df_5m.index else None
            current_1m = self.df_1m.loc[timestamp] if timestamp in self.df_1m.index else None
            
            if current_5m is None or current_1m is None:
                return False
            
            # Get ORB levels for the day
            orb_levels = self.calculate_orb(timestamp)
            if np.isnan(orb_levels['orb_high']):
                return False
            
            # Check if we're EXACTLY at the ORB window end (not before, not after)
            market_open = pd.Timestamp(datetime.combine(timestamp.date(), self.config.market_open))
            orb_end = market_open + timedelta(minutes=self.config.orb_minutes)
            
            if timestamp != orb_end:
                return False  # Only trade exactly at ORB end time
            
            # Check ORB breakout condition
            if direction == TradeDirection.LONG:
                orb_break = current_5m['close'] > orb_levels['orb_high']
            else:
                orb_break = current_5m['close'] < orb_levels['orb_low']
            
            if not orb_break:
                return False
            
            # Check 1-minute confirmation conditions
            if direction == TradeDirection.LONG:
                candle_bullish = current_1m['close'] > current_1m['open']
                rsi_condition = current_1m['rsi'] > 50
                ema_condition = current_1m['close'] > current_1m[f'ema_{self.config.ema_period}']
            else:
                candle_bullish = current_1m['close'] < current_1m['open']  # bearish for short
                rsi_condition = current_1m['rsi'] < 50
                ema_condition = current_1m['close'] < current_1m[f'ema_{self.config.ema_period}']
            
            # Volume filter (optional)
            volume_condition = True
            if self.config.use_volume_filter:
                volume_condition = current_1m['volume'] > (1.5 * current_1m['volume_ma'])
            
            # All conditions must be true
            return all([candle_bullish, rsi_condition, ema_condition, volume_condition])
            
        except (KeyError, IndexError):
            return False
            
    def calculate_confidence_score(self, timestamp: pd.Timestamp, direction: TradeDirection) -> Tuple[float, ConfidenceLevel]:
        """Enhanced confidence score with 5M base + 4H confluence for cleaner signals"""
        try:
            # Calculate base 5-minute confidence score (CLEANER SIGNALS)
            base_score = self._calculate_5m_confidence_score(timestamp, direction)
            
            # Add 4H confluence if enabled
            confluence_score = 0.0
            if self.config.use_4h_confluence:
                confluence_score = self._calculate_4h_confluence_score(timestamp, direction)
            
            total_score = base_score + confluence_score
            
            # Determine confidence level using enhanced thresholds
            if total_score >= self.config.confidence_high_threshold:
                confidence_level = ConfidenceLevel.HIGH
            elif total_score >= self.config.confidence_medium_threshold:
                confidence_level = ConfidenceLevel.MEDIUM
            else:
                confidence_level = ConfidenceLevel.LOW
                
            return total_score, confidence_level
            
        except (KeyError, IndexError):
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
            # SELECTIVE RSI SCORING - Quality zones only (max 3.0 points)
            if 50 <= rsi <= 65:
                score += 3.0  # SWEET SPOT - bullish with room to run
            elif 45 <= rsi < 50:
                score += 2.0  # Good momentum building
            elif 65 < rsi <= 75:
                score += 1.0  # Extended but possible
            elif 35 <= rsi < 45:
                score += 1.5  # Oversold bounce potential
            # rsi >75 or <35 = 0 points (too extreme)
                
            # SELECTIVE MACD SCORING - Must show real momentum (max 2.0 points)
            macd_hist = indicators_5m['macd_histogram_5m']
            if macd_hist > 0.05:  # Strong positive histogram
                score += 1.5
            elif macd_hist > 0:  # Weak positive histogram
                score += 0.5
                
            if indicators_5m['macd_5m'] > indicators_5m['macd_signal_5m']:
                # Only reward if it's a meaningful cross
                cross_strength = indicators_5m['macd_5m'] - indicators_5m['macd_signal_5m']
                if cross_strength > 0.02:
                    score += 1.0  # Strong bullish cross
                elif cross_strength > 0:
                    score += 0.5  # Weak bullish cross
                
            # VOLUME FILTER - Must have conviction (max 1.0 points)
            volume_ratio = indicators_5m['volume_5m'] / indicators_5m['volume_ma_5m']
            if volume_ratio > 1.5:
                score += 1.0  # High conviction
            elif volume_ratio > 1.1:
                score += 0.5  # Above average
            # Below average volume = 0 points
                
            # VWAP CONFIRMATION - Must be aligned (max 1.0 points)
            if indicators_5m['close_5m'] > indicators_5m['vwap_5m']:
                price_distance = (indicators_5m['close_5m'] - indicators_5m['vwap_5m']) / indicators_5m['vwap_5m'] * 100
                if price_distance > 0.1:  # Meaningful distance above VWAP
                    score += 1.0
                else:
                    score += 0.3  # Just barely above
                
        else:  # SHORT trades
            # SELECTIVE RSI SCORING - Quality zones only (max 3.0 points)
            if 35 <= rsi <= 50:
                score += 3.0  # SWEET SPOT - bearish with room to fall
            elif 50 < rsi <= 55:
                score += 2.0  # Good downward momentum building
            elif 25 <= rsi < 35:
                score += 1.0  # Extended down but possible
            elif 55 < rsi <= 65:
                score += 1.5  # Overbought fall potential
            # rsi <25 or >65 = 0 points (too extreme)
                
            # SELECTIVE MACD SCORING - Must show real momentum (max 2.0 points)
            macd_hist = indicators_5m['macd_histogram_5m']
            if macd_hist < -0.05:  # Strong negative histogram
                score += 1.5
            elif macd_hist < 0:  # Weak negative histogram
                score += 0.5
                
            if indicators_5m['macd_5m'] < indicators_5m['macd_signal_5m']:
                # Only reward if it's a meaningful cross
                cross_strength = indicators_5m['macd_signal_5m'] - indicators_5m['macd_5m']
                if cross_strength > 0.02:
                    score += 1.0  # Strong bearish cross
                elif cross_strength > 0:
                    score += 0.5  # Weak bearish cross
                
            # VOLUME FILTER - Must have conviction (max 1.0 points)
            volume_ratio = indicators_5m['volume_5m'] / indicators_5m['volume_ma_5m']
            if volume_ratio > 1.5:
                score += 1.0  # High conviction
            elif volume_ratio > 1.1:
                score += 0.5  # Above average
            # Below average volume = 0 points
                
            # VWAP CONFIRMATION - Must be aligned (max 1.0 points)
            if indicators_5m['close_5m'] < indicators_5m['vwap_5m']:
                price_distance = (indicators_5m['vwap_5m'] - indicators_5m['close_5m']) / indicators_5m['vwap_5m'] * 100
                if price_distance > 0.1:  # Meaningful distance below VWAP
                    score += 1.0
                else:
                    score += 0.3  # Just barely below
        
        return max(0.0, min(score, 7.0))  # Cap at 7.0, floor at 0 (can fail)
    
    def _calculate_1m_fallback_score(self, current_1m, direction: TradeDirection) -> float:
        """Selective fallback scoring using 1m data when 5m not available"""
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
        """Calculate 4H timeframe confluence score - LESS RESTRICTIVE OPTIONS SYSTEM (max 5.5 points)"""
        
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
            
    def calculate_trade_levels(self, entry_price: float, direction: TradeDirection, 
                            orb_levels: Dict[str, float], atr_value: float) -> Dict[str, float]:
        """Calculate stop loss and take profit levels for a trade"""
        orb_range = orb_levels['orb_range']
        
        if direction == TradeDirection.LONG:
            # Take profit levels using Fibonacci extensions
            tp1 = orb_levels['orb_low'] + (orb_range * 1.272)
            tp2 = orb_levels['orb_low'] + (orb_range * 1.618)
            tp3 = orb_levels['orb_low'] + (orb_range * 2.0)
            
            # Stop loss using ATR
            stop_loss = entry_price - (atr_value * self.config.atr_multiplier)
            
        else:  # SHORT
            # Take profit levels using Fibonacci extensions
            tp1 = orb_levels['orb_high'] - (orb_range * 1.272)
            tp2 = orb_levels['orb_high'] - (orb_range * 1.618)
            tp3 = orb_levels['orb_high'] - (orb_range * 2.0)
            
            # Stop loss using ATR
            stop_loss = entry_price + (atr_value * self.config.atr_multiplier)
        
        return {
            'stop_loss': stop_loss,
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3
        }
        
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
    
    def get_5m_indicators_at_time(self, timestamp: pd.Timestamp) -> Optional[Dict[str, float]]:
        """Get 5M indicators for the most recent 5M bar at/before timestamp"""
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

class ORBBacktester:
    """Opening Range Breakout backtesting engine"""
    
    def __init__(self, indicator: ORBIndicator, config: ORBConfig):
        self.indicator = indicator
        self.config = config
        self.trades: List[Trade] = []
        self.daily_pnl: Dict[str, float] = {}
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self.current_capital = config.initial_capital
        self.current_position: Optional[Trade] = None
        
    def calculate_position_size(self, entry_price: float, stop_loss: float, direction: TradeDirection) -> int:
        """Calculate position size based on risk per trade"""
        if direction == TradeDirection.LONG:
            risk_per_share = entry_price - stop_loss
        else:
            risk_per_share = stop_loss - entry_price
            
        if risk_per_share <= 0:
            return 0
            
        shares = int(self.config.risk_per_trade / risk_per_share)
        return max(1, shares)  # At least 1 share
        
    def calculate_dynamic_stop_multiplier(self, trade: Trade, current_time: pd.Timestamp) -> float:
        """Calculate current ATR multiplier based on time since entry"""
        # Determine narrowing start time
        if trade.narrowing_start_time is None:
            trade.narrowing_start_time = trade.entry_time + timedelta(minutes=self.config.narrowing_start_minutes)
        
        # Before narrowing starts, use initial multiplier
        if current_time < trade.narrowing_start_time:
            return self.config.initial_atr_multiplier
        
        # Calculate progress through narrowing period
        minutes_since_narrowing_start = (current_time - trade.narrowing_start_time).total_seconds() / 60
        total_narrowing_minutes = self.config.max_hold_minutes - self.config.narrowing_start_minutes
        
        if total_narrowing_minutes <= 0:
            return self.config.final_atr_multiplier
        
        progress_ratio = min(1.0, minutes_since_narrowing_start / total_narrowing_minutes)
        
        # Linear interpolation between initial and final multipliers
        multiplier_range = self.config.initial_atr_multiplier - self.config.final_atr_multiplier
        current_multiplier = self.config.initial_atr_multiplier - (multiplier_range * progress_ratio)
        
        return current_multiplier

    def update_stop_loss(self, trade: Trade, current_time: pd.Timestamp) -> float:
        """Update stop loss using dynamic narrowing algorithm"""
        try:
            # Calculate current dynamic stop multiplier
            current_multiplier = self.calculate_dynamic_stop_multiplier(trade, current_time)
            
            # Calculate new stop price based on current multiplier
            if trade.direction == TradeDirection.LONG:
                new_stop = trade.entry_price - (trade.atr_value * current_multiplier)
                # Ratchet rule: stop can only move up (more favorable)
                new_stop = max(trade.stop_loss, new_stop)
            else:  # SHORT
                new_stop = trade.entry_price + (trade.atr_value * current_multiplier)
                # Ratchet rule: stop can only move down (more favorable)
                new_stop = min(trade.stop_loss, new_stop)
            
            # Record stop history for visualization
            if abs(new_stop - trade.stop_loss) > 0.001:  # Only record if stop actually changed
                trade.stop_history.append((current_time, new_stop))
            
            return new_stop
            
        except Exception:
            # Fallback to current stop if calculation fails
            return trade.stop_loss
        
    def check_exit_conditions(self, trade: Trade, current_time: pd.Timestamp) -> Tuple[bool, ExitReason, float]:
        """Check if any exit conditions are met"""
        try:
            current_price = self.indicator.df_1m.loc[current_time, 'close']
            
            # Check time limit
            holding_time = (current_time - trade.entry_time).total_seconds() / 60
            if holding_time >= self.config.max_hold_minutes:
                return True, ExitReason.TIME_LIMIT, current_price
            
            if trade.direction == TradeDirection.LONG:
                # Check take profit levels
                if current_price >= trade.tp3:
                    return True, ExitReason.TP3, current_price
                elif current_price >= trade.tp2:
                    return True, ExitReason.TP2, current_price
                elif current_price >= trade.tp1:
                    return True, ExitReason.TP1, current_price
                
                # Check stop loss
                elif current_price <= trade.stop_loss:
                    return True, ExitReason.STOP_LOSS, current_price
                    
            else:  # SHORT
                # Check take profit levels
                if current_price <= trade.tp3:
                    return True, ExitReason.TP3, current_price
                elif current_price <= trade.tp2:
                    return True, ExitReason.TP2, current_price
                elif current_price <= trade.tp1:
                    return True, ExitReason.TP1, current_price
                
                # Check stop loss
                elif current_price >= trade.stop_loss:
                    return True, ExitReason.STOP_LOSS, current_price
                    
        except:
            pass
            
        return False, None, 0.0
        
    def update_mfe_mae(self, trade: Trade, current_time: pd.Timestamp):
        """Update Maximum Favorable/Adverse Excursion"""
        try:
            current_price = self.indicator.df_1m.loc[current_time, 'close']
            
            if trade.direction == TradeDirection.LONG:
                # For long positions
                favorable_move = current_price - trade.entry_price
                adverse_move = trade.entry_price - current_price
            else:
                # For short positions
                favorable_move = trade.entry_price - current_price
                adverse_move = current_price - trade.entry_price
            
            # Update MFE (Maximum Favorable Excursion)
            if favorable_move > trade.max_favorable_excursion:
                trade.max_favorable_excursion = favorable_move
                
            # Update MAE (Maximum Adverse Excursion)
            if adverse_move > trade.max_adverse_excursion:
                trade.max_adverse_excursion = adverse_move
                
        except:
            pass
            
    def execute_trade(self, timestamp: pd.Timestamp, direction: TradeDirection) -> bool:
        """Execute a trade if conditions are met"""
        # Check if we already have a position for today
        if self.config.one_trade_per_day and self.current_position is not None:
            if self.current_position.entry_time.date() == timestamp.date():
                return False
                
        # Get entry signal confirmation
        if not self.indicator.check_entry_signal(timestamp, direction):
            return False
            
        # Calculate confidence score
        confidence_score, confidence_level = self.indicator.calculate_confidence_score(timestamp, direction)
        
        # Apply confidence filter if enabled
        if self.config.use_confidence_filter:
            min_score_map = {
                ConfidenceLevel.LOW: 0.0,
                ConfidenceLevel.MEDIUM: 3.5,
                ConfidenceLevel.HIGH: 5.5
            }
            min_required = min_score_map[self.config.min_confidence_level]
            if confidence_score < min_required:
                return False
        
        try:
            # Get current price and ORB levels
            entry_price = self.indicator.df_1m.loc[timestamp, 'close']
            orb_levels = self.indicator.calculate_orb(timestamp)
            atr_value = self.indicator.get_nearest_5m_atr(timestamp)
            
            # Calculate initial dynamic stop loss
            if direction == TradeDirection.LONG:
                initial_stop = entry_price - (atr_value * self.config.initial_atr_multiplier)
            else:
                initial_stop = entry_price + (atr_value * self.config.initial_atr_multiplier)
            
            # Calculate trade levels using the initial stop
            trade_levels = self.indicator.calculate_trade_levels(
                entry_price, direction, orb_levels, atr_value
            )
            
            # Calculate position size based on initial wide stop
            shares = self.calculate_position_size(entry_price, initial_stop, direction)
            
            if shares == 0:
                return False
            
            # Create trade record with dynamic stop tracking
            trade = Trade(
                entry_time=timestamp,
                direction=direction,
                entry_price=entry_price,
                shares=shares,
                stop_loss=initial_stop,
                initial_stop_loss=initial_stop,
                atr_value=atr_value,
                tp1=trade_levels['tp1'],
                tp2=trade_levels['tp2'],
                tp3=trade_levels['tp3'],
                confidence_score=confidence_score,
                confidence_level=confidence_level,
                orb_high=orb_levels['orb_high'],
                orb_low=orb_levels['orb_low'],
                stop_history=[(timestamp, initial_stop)]
            )
            
            self.current_position = trade
            return True
            
        except Exception as e:
            print(f"Error executing trade: {e}")
            return False
            
    def close_position(self, exit_time: pd.Timestamp, exit_reason: ExitReason, exit_price: float):
        """Close the current position"""
        if self.current_position is None:
            return
            
        trade = self.current_position
        trade.exit_time = exit_time
        trade.exit_reason = exit_reason
        trade.exit_price = exit_price
        trade.holding_minutes = int((exit_time - trade.entry_time).total_seconds() / 60)
        
        # Calculate P&L
        if trade.direction == TradeDirection.LONG:
            trade.pnl = (exit_price - trade.entry_price) * trade.shares
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.shares
            
        trade.pnl_pct = (trade.pnl / (trade.entry_price * trade.shares)) * 100
        
        # Update capital
        self.current_capital += trade.pnl
        
        # Record daily P&L
        date_str = exit_time.date().strftime('%Y-%m-%d')
        if date_str not in self.daily_pnl:
            self.daily_pnl[date_str] = 0
        self.daily_pnl[date_str] += trade.pnl
        
        # Add to equity curve
        self.equity_curve.append((exit_time, self.current_capital))
        
        # Store completed trade
        self.trades.append(trade)
        self.current_position = None
        
    def run_backtest(self, start_date: str = None, end_date: str = None) -> Dict:
        """Run the complete backtest"""
        print("Starting ORB backtest...")
        
        # Filter data by date range if specified
        df_1m = self.indicator.df_1m
        if start_date:
            df_1m = df_1m[df_1m.index >= start_date]
        if end_date:
            df_1m = df_1m[df_1m.index <= end_date]
            
        # Initialize equity curve
        self.equity_curve = [(df_1m.index[0], self.config.initial_capital)]
        
        total_bars = len(df_1m)
        processed = 0
        
        # Main backtest loop - iterate through 1-minute bars
        for current_time in df_1m.index:
            processed += 1
            
            # Progress indicator
            if processed % 1000 == 0:
                progress = (processed / total_bars) * 100
                print(f"Progress: {progress:.1f}% - {current_time}")
            
            # Check if it's during market hours
            current_time_only = current_time.time()
            if current_time_only < self.config.market_open or current_time_only >= self.config.market_close:
                continue
            
            # If we have an open position, manage it
            if self.current_position is not None:
                # Update MFE/MAE
                self.update_mfe_mae(self.current_position, current_time)
                
                # Update dynamic stop loss based on frequency
                minutes_since_entry = (current_time - self.current_position.entry_time).total_seconds() / 60
                if (minutes_since_entry % self.config.stop_update_frequency_minutes) < 1.0:
                    self.current_position.stop_loss = self.update_stop_loss(self.current_position, current_time)
                
                # Check exit conditions
                should_exit, exit_reason, exit_price = self.check_exit_conditions(self.current_position, current_time)
                
                if should_exit:
                    self.close_position(current_time, exit_reason, exit_price)
                    continue
            
            # Look for new entry signals (only if no current position)
            if self.current_position is None:
                # Check if we're EXACTLY at the ORB window end (15 minutes after open)
                market_open = pd.Timestamp(datetime.combine(current_time.date(), self.config.market_open))
                orb_end = market_open + timedelta(minutes=self.config.orb_minutes)
                
                # Only check for entry at the exact ORB end time (9:45 AM for 15-min ORB)
                if current_time == orb_end:
                    # Try long entry first
                    if self.execute_trade(current_time, TradeDirection.LONG):
                        continue
                    
                    # If no long entry, try short entry
                    self.execute_trade(current_time, TradeDirection.SHORT)
        
        # Close any remaining position at the end
        if self.current_position is not None:
            final_time = df_1m.index[-1]
            final_price = df_1m.loc[final_time, 'close']
            self.close_position(final_time, ExitReason.TIME_LIMIT, final_price)
        
        print(f"Backtest completed. Total trades: {len(self.trades)}")
        
        # Generate performance summary
        return self.generate_performance_summary()
        
    def generate_performance_summary(self) -> Dict:
        """Generate comprehensive performance metrics"""
        if not self.trades:
            return {"error": "No trades executed"}
        
        trades_df = pd.DataFrame([{
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'direction': t.direction.value,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'pnl': t.pnl,
            'pnl_pct': t.pnl_pct,
            'exit_reason': t.exit_reason.value,
            'confidence_score': t.confidence_score,
            'confidence_level': t.confidence_level.value,
            'holding_minutes': t.holding_minutes,
            'mfe': t.max_favorable_excursion,
            'mae': t.max_adverse_excursion
        } for t in self.trades])
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 else float('inf')
        
        total_pnl = trades_df['pnl'].sum()
        total_return = (total_pnl / self.config.initial_capital) * 100
        
        # Drawdown calculation
        equity_values = [eq[1] for eq in self.equity_curve]
        running_max = np.maximum.accumulate(equity_values)
        drawdown = (equity_values - running_max) / running_max * 100
        max_drawdown = abs(min(drawdown))
        
        # Sharpe ratio (simplified - using trade returns)
        if len(trades_df) > 1:
            returns = trades_df['pnl_pct'] / 100
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
        else:
            sharpe_ratio = 0
        
        # Exit reason breakdown
        exit_reasons = trades_df['exit_reason'].value_counts()
        
        # Direction breakdown
        long_trades = trades_df[trades_df['direction'] == 'LONG']
        short_trades = trades_df[trades_df['direction'] == 'SHORT']
        
        # Confidence analysis
        confidence_performance = trades_df.groupby('confidence_level').agg({
            'pnl': ['count', 'mean', 'sum'],
            'pnl_pct': 'mean'
        }).round(2)
        
        return {
            'total_trades': total_trades,
            'win_rate': round(win_rate, 2),
            'profit_factor': round(profit_factor, 2),
            'total_pnl': round(total_pnl, 2),
            'total_return_pct': round(total_return, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'max_drawdown_pct': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'avg_holding_time': round(trades_df['holding_minutes'].mean(), 1),
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'long_win_rate': round(len(long_trades[long_trades['pnl'] > 0]) / len(long_trades) * 100, 2) if len(long_trades) > 0 else 0,
            'short_win_rate': round(len(short_trades[short_trades['pnl'] > 0]) / len(short_trades) * 100, 2) if len(short_trades) > 0 else 0,
            'exit_reasons': exit_reasons.to_dict(),
            'confidence_performance': confidence_performance,
            'best_trade': round(trades_df['pnl'].max(), 2),
            'worst_trade': round(trades_df['pnl'].min(), 2),
            'trades_df': trades_df
        }

class ORBVisualizer:
    """Visualization components for ORB trading system"""
    
    def __init__(self, backtester: ORBBacktester):
        self.backtester = backtester
        self.trades = backtester.trades
        self.equity_curve = backtester.equity_curve
        self.config = backtester.config
        
    def plot_equity_curve(self, figsize=(12, 6)):
        """Plot the equity curve"""
        if not self.equity_curve:
            print("No equity data to plot")
            return
            
        timestamps, equity_values = zip(*self.equity_curve)
        
        plt.figure(figsize=figsize)
        plt.plot(timestamps, equity_values, linewidth=2, color='blue')
        plt.title('Equity Curve - ORB Trading System', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Add initial capital line
        plt.axhline(y=self.config.initial_capital, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def plot_daily_pnl(self, figsize=(12, 6)):
        """Plot daily P&L distribution"""
        if not self.backtester.daily_pnl:
            print("No daily P&L data to plot")
            return
            
        dates = list(self.backtester.daily_pnl.keys())
        pnl_values = list(self.backtester.daily_pnl.values())
        
        plt.figure(figsize=figsize)
        colors = ['green' if pnl > 0 else 'red' for pnl in pnl_values]
        plt.bar(dates, pnl_values, color=colors, alpha=0.7)
        plt.title('Daily P&L Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Daily P&L ($)')
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.tight_layout()
        plt.show()
        
    def plot_win_loss_distribution(self, figsize=(10, 6)):
        """Plot win/loss distribution"""
        if not self.trades:
            print("No trades to analyze")
            return
            
        pnl_values = [trade.pnl for trade in self.trades]
        winning_trades = [pnl for pnl in pnl_values if pnl > 0]
        losing_trades = [pnl for pnl in pnl_values if pnl < 0]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Winning trades histogram
        if winning_trades:
            ax1.hist(winning_trades, bins=20, color='green', alpha=0.7, edgecolor='black')
            ax1.set_title('Winning Trades Distribution')
            ax1.set_xlabel('Profit ($)')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
        
        # Losing trades histogram  
        if losing_trades:
            ax2.hist(losing_trades, bins=20, color='red', alpha=0.7, edgecolor='black')
            ax2.set_title('Losing Trades Distribution')
            ax2.set_xlabel('Loss ($)')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Win/Loss Distribution Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    def plot_confidence_analysis(self, figsize=(12, 8)):
        """Plot confidence score analysis"""
        if not self.trades:
            print("No trades to analyze")
            return
            
        confidence_data = {
            'HIGH': {'wins': 0, 'losses': 0, 'total_pnl': 0, 'count': 0},
            'MEDIUM': {'wins': 0, 'losses': 0, 'total_pnl': 0, 'count': 0},
            'LOW': {'wins': 0, 'losses': 0, 'total_pnl': 0, 'count': 0}
        }
        
        for trade in self.trades:
            level = trade.confidence_level.value
            confidence_data[level]['count'] += 1
            confidence_data[level]['total_pnl'] += trade.pnl
            
            if trade.pnl > 0:
                confidence_data[level]['wins'] += 1
            else:
                confidence_data[level]['losses'] += 1
        
        levels = list(confidence_data.keys())
        win_rates = [confidence_data[level]['wins'] / max(confidence_data[level]['count'], 1) * 100 for level in levels]
        avg_pnls = [confidence_data[level]['total_pnl'] / max(confidence_data[level]['count'], 1) for level in levels]
        trade_counts = [confidence_data[level]['count'] for level in levels]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Win rates by confidence
        ax1.bar(levels, win_rates, color=['red', 'yellow', 'green'], alpha=0.7)
        ax1.set_title('Win Rate by Confidence Level')
        ax1.set_ylabel('Win Rate (%)')
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
        
        # Average P&L by confidence
        colors = ['green' if pnl > 0 else 'red' for pnl in avg_pnls]
        ax2.bar(levels, avg_pnls, color=colors, alpha=0.7)
        ax2.set_title('Average P&L by Confidence Level')
        ax2.set_ylabel('Average P&L ($)')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Trade count by confidence
        ax3.bar(levels, trade_counts, color=['red', 'yellow', 'green'], alpha=0.7)
        ax3.set_title('Trade Count by Confidence Level')
        ax3.set_ylabel('Number of Trades')
        ax3.grid(True, alpha=0.3)
        
        # Cumulative P&L by confidence
        cumulative_pnls = [confidence_data[level]['total_pnl'] for level in levels]
        colors = ['green' if pnl > 0 else 'red' for pnl in cumulative_pnls]
        ax4.bar(levels, cumulative_pnls, color=colors, alpha=0.7)
        ax4.set_title('Total P&L by Confidence Level')
        ax4.set_ylabel('Total P&L ($)')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.suptitle('Confidence Score Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    def plot_entry_time_analysis(self, figsize=(12, 6)):
        """Analyze entry times and their performance"""
        if not self.trades:
            print("No trades to analyze")
            return
            
        # Extract entry hours
        entry_hours = [trade.entry_time.hour + trade.entry_time.minute/60 for trade in self.trades]
        entry_pnls = [trade.pnl for trade in self.trades]
        
        # Create hourly buckets
        hourly_data = {}
        for hour, pnl in zip(entry_hours, entry_pnls):
            hour_bucket = int(hour * 2) / 2  # 30-minute buckets
            if hour_bucket not in hourly_data:
                hourly_data[hour_bucket] = []
            hourly_data[hour_bucket].append(pnl)
        
        hours = sorted(hourly_data.keys())
        avg_pnls = [np.mean(hourly_data[hour]) for hour in hours]
        trade_counts = [len(hourly_data[hour]) for hour in hours]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Average P&L by entry time
        colors = ['green' if pnl > 0 else 'red' for pnl in avg_pnls]
        ax1.bar(hours, avg_pnls, width=0.4, color=colors, alpha=0.7)
        ax1.set_title('Average P&L by Entry Time')
        ax1.set_ylabel('Average P&L ($)')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Trade count by entry time
        ax2.bar(hours, trade_counts, width=0.4, color='blue', alpha=0.7)
        ax2.set_title('Trade Count by Entry Time')
        ax2.set_xlabel('Entry Time (Hours)')
        ax2.set_ylabel('Number of Trades')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Entry Time Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    def plot_sample_trade(self, trade_index: int = 0, figsize=(15, 8)):
        """Plot a sample trade with entry, exit, and ORB levels"""
        if not self.trades or trade_index >= len(self.trades):
            print("Invalid trade index or no trades available")
            return
            
        trade = self.trades[trade_index]
        
        # Get data for the trading day
        trade_date = trade.entry_time.date()
        day_data_1m = self.backtester.indicator.df_1m[
            self.backtester.indicator.df_1m['date_only'] == trade_date
        ]
        
        if day_data_1m.empty:
            print(f"No data available for {trade_date}")
            return
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                      gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        
        # Price chart
        ax1.plot(day_data_1m.index, day_data_1m['close'], linewidth=1, color='black', label='Price')
        
        # ORB levels
        ax1.axhline(y=trade.orb_high, color='blue', linestyle='--', alpha=0.7, label='ORB High')
        ax1.axhline(y=trade.orb_low, color='blue', linestyle='--', alpha=0.7, label='ORB Low')
        
        # Trade levels
        ax1.axhline(y=trade.entry_price, color='orange', linestyle='-', alpha=0.8, label='Entry')
        ax1.axhline(y=trade.tp1, color='green', linestyle=':', alpha=0.7, label='TP1')
        ax1.axhline(y=trade.tp2, color='green', linestyle=':', alpha=0.7, label='TP2')
        ax1.axhline(y=trade.tp3, color='green', linestyle=':', alpha=0.7, label='TP3')
        ax1.axhline(y=trade.stop_loss, color='red', linestyle=':', alpha=0.7, label='Stop Loss')
        
        # Entry and exit markers
        ax1.scatter([trade.entry_time], [trade.entry_price], color='green', s=100, marker='^', 
                   label='Entry', zorder=5)
        if trade.exit_time:
            ax1.scatter([trade.exit_time], [trade.exit_price], color='red', s=100, marker='v',
                       label='Exit', zorder=5)
        
        # ORB window shading
        market_open = pd.Timestamp(datetime.combine(trade_date, self.config.market_open))
        orb_end = market_open + timedelta(minutes=self.config.orb_minutes)
        ax1.axvspan(market_open, orb_end, alpha=0.2, color='yellow', label='ORB Window')
        
        ax1.set_title(f'Sample Trade - {trade_date} | Direction: {trade.direction.value} | '
                     f'P&L: ${trade.pnl:.2f} | Exit: {trade.exit_reason.value if trade.exit_reason else "N/A"}')
        ax1.set_ylabel('Price ($)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Volume chart
        ax2.bar(day_data_1m.index, day_data_1m['volume'], alpha=0.6, color='gray')
        ax2.set_title('Volume')
        ax2.set_ylabel('Volume')
        ax2.set_xlabel('Time')
        ax2.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Print trade details
        print(f"\nTrade Details:")
        print(f"Entry Time: {trade.entry_time}")
        print(f"Exit Time: {trade.exit_time}")
        print(f"Direction: {trade.direction.value}")
        print(f"Entry Price: ${trade.entry_price:.2f}")
        print(f"Exit Price: ${trade.exit_price:.2f}")
        print(f"Shares: {trade.shares}")
        print(f"P&L: ${trade.pnl:.2f} ({trade.pnl_pct:.2f}%)")
        print(f"Exit Reason: {trade.exit_reason.value if trade.exit_reason else 'N/A'}")
        print(f"Confidence: {trade.confidence_level.value} ({trade.confidence_score:.1f})")
        print(f"Holding Time: {trade.holding_minutes} minutes")
        print(f"Max Favorable Excursion: ${trade.max_favorable_excursion:.2f}")
        print(f"Max Adverse Excursion: ${trade.max_adverse_excursion:.2f}")
        
    def generate_comprehensive_report(self):
        """Generate a comprehensive visual report"""
        print("=" * 60)
        print("ORB TRADING SYSTEM - COMPREHENSIVE REPORT")
        print("=" * 60)
        
        # Performance summary
        performance = self.backtester.generate_performance_summary()
        
        print(f"\nPERFORMANCE SUMMARY:")
        print(f"Total Trades: {performance['total_trades']}")
        print(f"Win Rate: {performance['win_rate']:.2f}%")
        print(f"Profit Factor: {performance['profit_factor']:.2f}")
        print(f"Total Return: {performance['total_return_pct']:.2f}%")
        print(f"Total P&L: ${performance['total_pnl']:,.2f}")
        print(f"Average Win: ${performance['avg_win']:,.2f}")
        print(f"Average Loss: ${performance['avg_loss']:,.2f}")
        print(f"Max Drawdown: {performance['max_drawdown_pct']:.2f}%")
        print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        print(f"Average Holding Time: {performance['avg_holding_time']:.1f} minutes")
        
        print(f"\nTRADE BREAKDOWN:")
        print(f"Long Trades: {performance['long_trades']} (Win Rate: {performance['long_win_rate']:.2f}%)")
        print(f"Short Trades: {performance['short_trades']} (Win Rate: {performance['short_win_rate']:.2f}%)")
        
        print(f"\nEXIT REASONS:")
        for reason, count in performance['exit_reasons'].items():
            print(f"{reason}: {count} trades")
        
        print(f"\nBEST/WORST TRADES:")
        print(f"Best Trade: ${performance['best_trade']:,.2f}")
        print(f"Worst Trade: ${performance['worst_trade']:,.2f}")
        
        # Generate all visualizations
        print(f"\nGenerating visualizations...")
        
        self.plot_equity_curve()
        self.plot_daily_pnl()
        self.plot_win_loss_distribution()
        self.plot_confidence_analysis()
        self.plot_entry_time_analysis()
        
        # Show sample trades
        if len(self.trades) > 0:
            print(f"\nShowing sample trades...")
            self.plot_sample_trade(0)  # First trade
            if len(self.trades) > 1:
                self.plot_sample_trade(-1)  # Last trade

def create_orb_system(df_1m: pd.DataFrame, df_5m: pd.DataFrame, df_4h: pd.DataFrame,
                     config: ORBConfig = None) -> Tuple[ORBIndicator, ORBBacktester, ORBVisualizer]:
    """Factory function to create a complete ORB trading system with 4H confluence support"""
    if config is None:
        config = ORBConfig()
    
    # Create indicator with 4H data
    indicator = ORBIndicator(df_1m, df_5m, df_4h, config)
    
    # Create backtester
    backtester = ORBBacktester(indicator, config)
    
    # Create visualizer
    visualizer = ORBVisualizer(backtester)
    
    return indicator, backtester, visualizer

def run_orb_analysis(df_1m: pd.DataFrame, df_5m: pd.DataFrame, df_4h: pd.DataFrame,
                    config: ORBConfig = None, start_date: str = None, 
                    end_date: str = None) -> Dict:
    """Complete ORB analysis workflow with 4H confluence support"""
    print("Initializing ORB Trading System with 4H confluence...")
    
    # Create system components
    indicator, backtester, visualizer = create_orb_system(df_1m, df_5m, df_4h, config)
    
    # Run backtest
    print("Running backtest...")
    results = backtester.run_backtest(start_date, end_date)
    
    # Generate comprehensive report
    visualizer.generate_comprehensive_report()
    
    return results