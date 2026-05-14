# Strategy: 6h_EMA50_Vol_LowVol_Filter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.061 | +22.0% | -14.5% | 62 | PASS |
| ETHUSDT | -0.192 | +4.4% | -23.5% | 64 | FAIL |
| SOLUSDT | 0.191 | +31.1% | -39.1% | 47 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.211 | +8.4% | -5.8% | 20 | PASS |
| SOLUSDT | 0.236 | +9.8% | -12.6% | 18 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for calculations (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d ATR(14) for volatility
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d_arr, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Volume filter: volume > 1.8x 30-period average (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    # Volatility filter: ATR below its 50-period median (low volatility regime)
    atr_median = pd.Series(atr14_1d_aligned).rolling(window=50, min_periods=14).median().values
    vol_filter = atr14_1d_aligned < atr_median
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_median[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above EMA50 + volume filter + low volatility
            if (close[i] > ema50_1d_aligned[i] and 
                volume_filter[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below EMA50 + volume filter + low volatility
            elif (close[i] < ema50_1d_aligned[i] and 
                  volume_filter[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below EMA50 (trend change)
            if close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above EMA50 (trend change)
            if close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_EMA50_Vol_LowVol_Filter_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-27 18:16
