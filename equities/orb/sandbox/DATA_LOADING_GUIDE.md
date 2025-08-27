# 📊 QQQ Data Loading System - No More Timeouts!

## 🎯 Quick Start

### Step 1: Backfill Data (One-time setup)
```bash
cd /Users/mmirandi/Desktop/Finance
python run_backfill.py
```

This creates 3 CSV files:
- `qqq_1m_backfill.csv` (14 days of 1-minute data)
- `qqq_5m_backfill.csv` (30 days of 5-minute data)  
- `qqq_4h_backfill.csv` (90 days of 4-hour data)

### Step 2: Use in Your Notebooks
Your sandbox.ipynb now automatically loads from CSV files - no more timeouts!

---

## 🔧 Usage Options

### Option 1: CSV Only (Recommended - Fast & Reliable)
```python
from qqq_data_loader import quick_csv_load
df_1m, df_5m, df_4h = quick_csv_load()
```

### Option 2: Smart Loading (Live → CSV Fallback)
```python  
from qqq_data_loader import get_qqq_data
df_1m, df_5m, df_4h = get_qqq_data(prefer_live=True, live_days=2)
```

### Option 3: Test Live Connection
```python
from qqq_data_loader import quick_live_test
if quick_live_test():
    print("Live connection working!")
```

---

## 📅 Data Coverage

| **Timeframe** | **Period** | **Bars** | **Use Case** |
|---------------|------------|----------|--------------|
| 1-minute | 14 days | ~6,000 | ORB signals, entry timing |
| 5-minute | 30 days | ~2,500 | ATR calculation, levels |
| 4-hour | 90 days | ~350 | RSI/MACD confluence |

---

## 🔄 Updating Data

### Manual Update (Run when needed)
```bash
python run_backfill.py
```

### Automatic Update (Advanced)
Add to crontab for daily updates:
```bash
0 18 * * * cd /Users/mmirandi/Desktop/Finance && python run_backfill.py
```

---

## 🚨 Troubleshooting

### If Backfill Fails:
1. **Check IBKR Connection**: Ensure TWS/Gateway is running
2. **Reduce Chunk Size**: Edit `run_backfill.py` - smaller chunks
3. **Add More Delays**: Increase sleep times between requests
4. **Check Permissions**: Ensure you have historical data permissions

### If CSV Loading Fails:
```python
import os
files = ['qqq_1m_backfill.csv', 'qqq_5m_backfill.csv', 'qqq_4h_backfill.csv']
for f in files:
    print(f"{f}: {'✅' if os.path.exists(f) else '❌'}")
```

---

## ⚡ Performance Tips

1. **Use CSV Loading**: Fastest and most reliable for development
2. **Backfill During Off-Hours**: Less IBKR load = fewer timeouts
3. **Keep CSV Files**: Don't delete them - they're your fallback
4. **Monitor File Sizes**: 1m should be ~1MB, 5m ~500KB, 4h ~50KB

---

## 🎯 Integration with ORB System

Your enhanced ORB system now gets:
- ✅ **1-minute data**: For precise entry signals
- ✅ **5-minute data**: For ATR and trade levels  
- ✅ **4-hour data**: For confluence analysis
- ✅ **No timeouts**: Reliable CSV fallback
- ✅ **Historical depth**: Sufficient for all indicators

The system automatically handles timezone conversion and data validation!