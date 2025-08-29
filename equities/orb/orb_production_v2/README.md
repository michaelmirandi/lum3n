# ORB Production V2 - Fast & Clean

## 🎯 Purpose
Clean, fast implementation focused on:
- **Rapid historical data loading**
- **Efficient real-time streaming** 
- **Correct candle timing** (timestamp = period START)
- **Minimal complexity**

## 🏗️ Architecture

```
orb_production_v2/
├── config.yaml          # Simple configuration
├── main.py              # Main entry point
├── test_timing.py       # Verify candle timing
├── data/
│   ├── historical.py    # Fast historical data fetch
│   └── streamer.py      # Live streaming + aggregation
```

## 📊 Key Features

### Candle Timing (CRITICAL)
- **Timestamp = Period START time**
- `14:39` 1m candle = `14:39:00` to `14:39:59`
- `14:40` 5m candle = `14:40:00` to `14:44:59`
- This ensures ORB indicators calculate correctly

### Historical Data
- Fast parallel fetch for 1m, 5m, 4h timeframes
- Configurable history depth (default 5 days)
- Proper timezone handling (Eastern)

### Live Streaming  
- 5-second bar aggregation into 1m, 5m, 4h
- Memory-efficient circular buffers
- Single finalization per candle (no duplicates)
- Real-time console logging

## 🚀 Usage

```bash
# Run main system
python main.py

# Test candle timing
python test_timing.py
```

## ⚙️ Configuration

```yaml
# config.yaml
orb_minutes: 20
symbols: ["QQQ"]
historical_days: 5
buffer_size: 500
ibkr_host: "127.0.0.1"
ibkr_port: 7496
```

## 🎪 Example Output

```
📚 Loading historical data...
✅ QQQ: 1950 1m, 390 5m, 120 4h bars

📡 Starting live stream...
✅ Streaming QQQ

🔴 LIVE MONITORING
1m  | 14:39 | QQQ | C:$423.45 V:12500
5m  | 14:40 | QQQ | C:$423.78 V:98750  
```

## 🔧 Next Steps
- Add ORB calculation logic
- Add signal detection
- Add position management