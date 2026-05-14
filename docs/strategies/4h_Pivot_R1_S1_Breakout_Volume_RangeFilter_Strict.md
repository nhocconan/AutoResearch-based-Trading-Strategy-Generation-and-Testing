# Strategy: 4h_Pivot_R1_S1_Breakout_Volume_RangeFilter_Strict

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.381 | +38.5% | -10.6% | 295 | PASS |
| ETHUSDT | 0.803 | +71.9% | -10.5% | 260 | PASS |
| SOLUSDT | 0.741 | +94.4% | -15.2% | 205 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.952 | -10.6% | -12.9% | 114 | FAIL |
| ETHUSDT | 0.744 | +16.7% | -6.1% | 91 | PASS |
| SOLUSDT | 0.419 | +11.9% | -10.0% | 74 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Pivot_R1_S1_Breakout_Volume_RangeFilter_Strict"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1_1d = close_1d + range_1d * 1.1 / 12.0
    s1_1d = close_1d - range_1d * 1.1 / 12.0
    
    # Align Camarilla levels to 4h timeframe
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 4h ATR for volatility and stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_4h = pd.Series(tr).rolling(window=15, min_periods=15).mean().values
    
    # Volume confirmation: current volume > 2.2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60
    
    for i in range(start_idx, n):
        if np.isnan(pivot_4h[i]) or np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or \
           np.isnan(atr_4h[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr = atr_4h[i]
        pivot = pivot_4h[i]
        r1 = r1_4h[i]
        s1 = s1_4h[i]
        
        volume_confirmed = vol > 2.2 * vol_ma
        
        if position == 0:
            # Long: Price breaks above R1 + volume
            if price > r1 and volume_confirmed:
                signals[i] = 0.28
                position = 1
            # Short: Price breaks below S1 + volume
            elif price < s1 and volume_confirmed:
                signals[i] = -0.28
                position = -1
        
        elif position == 1:
            # Exit: Price returns below pivot OR ATR stop (2.5x ATR from entry)
            if price < pivot or price < (high[i] - 2.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        
        elif position == -1:
            # Exit: Price returns above pivot OR ATR stop (2.5x ATR from entry)
            if price > pivot or price > (low[i] + 2.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals
```

## Last Updated
2026-04-19 10:07
