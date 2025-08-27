# ORB Production System - Phase 2: Core ORB Logic

## Overview
Phase 2 implements the core ORB trading logic and technical indicator calculations:
- Direct port of ORBConfig and ORBIndicator classes
- All technical indicators (RSI, MACD, ATR, VWAP, etc.)
- ORB level calculations (20-minute opening range)
- Confidence scoring system with 4H confluence
- Trade level calculations (stops, take profits)

## Key Components

### 1. ORBConfig
Configuration class containing all trading parameters:
- ORB duration (20 minutes default)
- Dynamic stop loss parameters 
- 4H confluence settings
- Confidence thresholds

### 2. ORBIndicator  
Main indicator class with methods:
- `calculate_orb()` - Opening range levels
- `check_entry_signal()` - Entry conditions
- `calculate_confidence_score()` - Signal quality scoring
- `calculate_trade_levels()` - Stop/TP calculations

### 3. Technical Indicators
All indicators from proven backtesting system:
- **1m timeframe**: RSI, EMA(9), MACD, Volume MA, VWAP
- **5m timeframe**: RSI, EMA(21), MACD, ATR, Volume MA, VWAP  
- **4h timeframe**: RSI, MACD, EMA(20) for confluence

## Testing

Run Phase 2 tests to validate all components:

```bash
python test_phase2.py
```

### Test Coverage:

1. **ORB Calculations** - Validates 20-minute opening range calculations
2. **Technical Indicators** - Confirms all indicators calculate correctly
3. **Confidence Scoring** - Tests 5M base + 4H confluence scoring
4. **Entry Signal Detection** - Validates breakout detection logic
5. **Trade Level Calculations** - Tests stop loss and take profit levels

## Key Features Ported

### From orb_trading_system.py:
✅ **Exact ORB calculation** (20-minute opening range)  
✅ **All technical indicators** with identical formulas  
✅ **4H confluence system** for signal filtering  
✅ **Dynamic confidence scoring** (max 13 points)  
✅ **Trade level calculations** using Fibonacci extensions  

### Confidence Scoring System:
- **5M Base Score**: RSI zones, MACD momentum, volume, VWAP (max 7 pts)
- **4H Confluence**: RSI alignment, MACD direction (max 5.5 pts) 
- **Total Possible**: 12.5 points
- **HIGH Threshold**: 6.0+ points
- **MEDIUM Threshold**: 4.0+ points

### ORB Breakout Rules:
- **Entry Window**: 9:50 AM - 10:30 AM (after ORB completion, within 60 min of open)
- Price must break above ORB high (LONG) or below ORB low (SHORT)
- First valid breakout wins (one trade per day)
- 1m candle confirmation required
- Optional volume filter (1.5x volume MA)

## Expected Test Results:

✅ **ORB Calculations**: 80%+ success rate on recent trading days  
✅ **Technical Indicators**: >90% data coverage for all indicators  
✅ **Confidence Scoring**: Valid scores (0-13 range) with 4H confluence  
✅ **Trade Levels**: Properly ordered stops and take profits  

## Integration with Phase 1:

Phase 2 seamlessly integrates with Phase 1's data pipeline:
- Uses cached historical data for indicator calculations
- Works with live streaming DataFrames
- Maintains timezone consistency
- Handles missing/incomplete data gracefully

## Next Steps (Phase 3):

After Phase 2 validation:
- Signal detection layer
- Real-time signal generation
- Position management integration
- Live trading preparation

## File Structure:
```
orb_live/core/
├── __init__.py
└── orb_system.py      # Complete ORB system with all classes

tests/
└── test_phase2.py     # Comprehensive validation tests
```

The Phase 2 implementation maintains 100% fidelity to your proven backtesting system while adapting it for live trading integration.