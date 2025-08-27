# ORB Production System - Phase 1: Data Pipeline

## Overview
Phase 1 implements the core data pipeline for the ORB trading system:
- IBKR connection and live data streaming
- 5-second bar aggregation into 1m, 5m, 15m, 4h candles
- Historical data caching for technical indicators
- DataFrame buffer management

## Setup

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Configure IBKR Connection
- Ensure IB Gateway or TWS is running
- Default port: 7496 (live data)
- Note: Paper trading account does NOT stream live data

### 3. Test the System
```bash
python test_phase1.py
```

## Components

### Data Flow
1. **Historical Cache** (`cache.py`)
   - Fetches previous days' data on startup
   - Ensures indicators have sufficient data
   - Caches to disk for faster restarts

2. **Candle Aggregator** (`aggregator.py`)
   - Receives 5-second bars from IBKR
   - Aggregates into 1m, 5m, 15m, 4h candles
   - Maintains rolling buffers
   - Color-coded console output

3. **Live Stream** (`stream.py`)
   - Manages IBKR connection
   - Combines historical + live data
   - Updates DataFrames in real-time

## Testing

The test script runs 4 tests:

1. **Connection Test**: Verifies IBKR connection
2. **Historical Data Test**: Checks cache loading
3. **Live Streaming Test**: Streams for 60 seconds
4. **Integration Test**: Full 2-minute test

## What You Should See

When running correctly:
- Colored bars in console (green = up, red = down)
- 5s bars streaming every 5 seconds
- 1m candles completing every minute
- 5m candles every 5 minutes
- Historical data loaded showing 100+ bars per timeframe

## Directory Structure
```
orb_production/
├── cache/                  # Historical data cache files
├── logs/                   # Log files (future)
├── orb_live/
│   ├── data/
│   │   ├── aggregator.py  # Candle aggregation logic
│   │   ├── cache.py       # Historical data management
│   │   └── stream.py      # Main streaming controller
│   └── ...
└── test_phase1.py         # Test script
```

## Next Steps (Phase 2)
- Port ORB calculation logic from orb_trading_system.py
- Implement technical indicators (RSI, MACD, ATR, etc.)
- Add confidence scoring system

## Troubleshooting

### "Cannot connect to IBKR"
- Check IB Gateway/TWS is running
- Verify port 7496 is correct
- Check API connections are enabled in IB Gateway

### "No data streaming"
- Ensure you're using LIVE account (paper has no streaming)
- Market hours: Best tested during market hours
- Symbol must be valid (default: QQQ)

### "Insufficient historical data"
- First run fetches from IBKR (may take 30 seconds)
- Subsequent runs use cache (instant)
- Cache stored in `cache/` directory

## Important Notes

1. **No Paper Trading**: IBKR paper accounts don't stream data
2. **Market Hours**: Best tested during market hours for live data
3. **Cache Management**: Delete `cache/` folder to force fresh data fetch
4. **Buffer Size**: Keeps 390 minutes (full trading day) of 1m data