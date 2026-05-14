# Strategy: 4h_1d_camarilla_breakout_v4

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.296 | +28.7% | -7.4% | 75 | KEEP |
| ETHUSDT | 0.536 | +36.5% | -5.0% | 57 | KEEP |
| SOLUSDT | -0.009 | +20.2% | -9.9% | 39 | DISCARD |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.171 | -5.6% | -5.6% | 34 | DISCARD |
| ETHUSDT | 0.528 | +10.0% | -5.3% | 25 | KEEP |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v4"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla levels
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (using prior day's OHLC)
    r4 = np.full(len(df_d), np.nan)
    s4 = np.full(len(df_d), np.nan)
    prev_high = np.full(len(df_d), np.nan)
    prev_low = np.full(len(df_d), np.nan)
    for i in range(1, len(df_d)):
        ph = float(df_d['high'].iloc[i-1])
        pl = float(df_d['low'].iloc[i-1])
        pc = float(df_d['close'].iloc[i-1])
        r4[i] = pc + (ph - pl) * 1.1 / 2
        s4[i] = pc - (ph - pl) * 1.1 / 2
        prev_high[i] = ph
        prev_low[i] = pl
    
    # Align daily values to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_d, s4)
    prev_high_aligned = align_htf_to_ltf(prices, df_d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_d, prev_low)
    
    # Volume confirmation: 3-period average (12h)
    vol_ma_3 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 3:
            vol_sum -= volume[i-3]
        if i >= 2:
            vol_ma_3[i] = vol_sum / 3
    
    # Choppiness regime filter (14-period)
    chop = np.full(n, np.nan)
    for i in range(n):
        if i >= 13:
            high_max = np.max(high[i-13:i+1])
            low_min = np.min(low[i-13:i+1])
            sum_true_range = 0.0
            for j in range(14):
                tr = max(high[i-j] - low[i-j], 
                         abs(high[i-j] - close[i-j-1]) if i-j-1 >= 0 else high[i-j] - low[i-j],
                         abs(low[i-j] - close[i-j-1]) if i-j-1 >= 0 else high[i-j] - low[i-j])
                sum_true_range += tr
            if sum_true_range > 0:
                chop[i] = 100 * np.log10(sum_true_range / (high_max - low_min)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(prev_high_aligned[i]) or 
            np.isnan(prev_low_aligned[i]) or 
            np.isnan(vol_ma_3[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes back inside previous day's range OR chop > 61.8 (trending ends)
            if (close[i] <= prev_high_aligned[i] and close[i] >= prev_low_aligned[i]) or chop[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes back inside previous day's range OR chop > 61.8
            if (close[i] <= prev_high_aligned[i] and close[i] >= prev_low_aligned[i]) or chop[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above R4 with volume confirmation AND chop < 61.8 (not too choppy)
            vol_ratio = volume[i] / vol_ma_3[i] if vol_ma_3[i] > 0 else 0
            if (close[i] > r4_aligned[i] and 
                vol_ratio > 2.0 and 
                chop[i] < 61.8):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below S4 with volume confirmation AND chop < 61.8
            elif (close[i] < s4_aligned[i] and 
                  vol_ratio > 2.0 and 
                  chop[i] < 61.8):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-12 12:42
