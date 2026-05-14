# Strategy: 4h_1d_camarilla_breakout_v30

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.386 | +33.5% | -13.4% | 150 | PASS |
| ETHUSDT | 0.484 | +40.3% | -6.4% | 116 | PASS |
| SOLUSDT | 0.288 | +36.0% | -13.6% | 112 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.337 | -3.4% | -5.8% | 66 | FAIL |
| ETHUSDT | 1.110 | +18.3% | -8.3% | 56 | PASS |
| SOLUSDT | -0.003 | +5.8% | -13.9% | 35 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with volume confirmation and choppiness filter
# Works in bull/bear by using mean-reversion at extreme levels (S4/R4) with volume confirmation
# Target: 20-40 trades/year to avoid fee drag, focusing on high-probability breakouts

name = "4h_1d_camarilla_breakout_v30"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (using prior day's OHLC)
    r4 = np.full(len(df_1d), np.nan)
    s4 = np.full(len(df_1d), np.nan)
    prev_high = np.full(len(df_1d), np.nan)
    prev_low = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        ph = float(df_1d['high'].iloc[i-1])
        pl = float(df_1d['low'].iloc[i-1])
        pc = float(df_1d['close'].iloc[i-1])
        r4[i] = pc + (ph - pl) * 1.1 / 2
        s4[i] = pc - (ph - pl) * 1.1 / 2
        prev_high[i] = ph
        prev_low[i] = pl
    
    # Align 1d values to 4h timeframe
    r4_4h = align_htf_to_ltf(prices, df_1d, r4)
    s4_4h = align_htf_to_ltf(prices, df_1d, s4)
    prev_high_4h = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_4h = align_htf_to_ltf(prices, df_1d, prev_low)
    
    # Volume confirmation: 4-period average (16h)
    vol_ma_4 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 4:
            vol_sum -= volume[i-4]
        if i >= 3:
            vol_ma_4[i] = vol_sum / 4
    
    # Choppiness regime filter (14-period) - optimized version
    chop = np.full(n, np.nan)
    for i in range(13, n):
        high_max = np.max(high[i-13:i+1])
        low_min = np.min(low[i-13:i+1])
        if high_max <= low_min:
            chop[i] = np.nan
            continue
        sum_true_range = 0.0
        for j in range(14):
            idx = i - j
            tr = high[idx] - low[idx]
            if idx > 0:
                tr = max(tr, abs(high[idx] - close[idx-1]), abs(low[idx] - close[idx-1]))
            sum_true_range += tr
        if sum_true_range > 0:
            chop[i] = 100 * np.log10(sum_true_range / (high_max - low_min)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(r4_4h[i]) or 
            np.isnan(s4_4h[i]) or 
            np.isnan(prev_high_4h[i]) or 
            np.isnan(prev_low_4h[i]) or 
            np.isnan(vol_ma_4[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes back inside previous day's range OR chop > 61.8 (trending ends)
            if (close[i] <= prev_high_4h[i] and close[i] >= prev_low_4h[i]) or chop[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes back inside previous day's range OR chop > 61.8
            if (close[i] <= prev_high_4h[i] and close[i] >= prev_low_4h[i]) or chop[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above R4 with volume confirmation AND chop < 61.8 (not too choppy)
            vol_ratio = volume[i] / vol_ma_4[i] if vol_ma_4[i] > 0 else 0
            if (close[i] > r4_4h[i] and 
                vol_ratio > 2.0 and 
                chop[i] < 61.8):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below S4 with volume confirmation AND chop < 61.8
            elif (close[i] < s4_4h[i] and 
                  vol_ratio > 2.0 and 
                  chop[i] < 61.8):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-09 10:27
