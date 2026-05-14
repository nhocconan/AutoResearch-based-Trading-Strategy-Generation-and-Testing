# Strategy: 6h_1d_camarilla_breakout_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.268 | +14.4% | -7.0% | 154 | FAIL |
| ETHUSDT | 0.298 | +31.8% | -6.0% | 116 | PASS |
| SOLUSDT | -0.134 | +12.6% | -16.3% | 86 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.205 | +7.9% | -6.7% | 49 | PASS |

## Code
```python
# 6h Camarilla Breakout with Volume Confirmation
# Hypothesis: Breakouts beyond daily Camarilla R4/S4 levels with volume confirmation capture
# strong momentum moves. Works in both bull/bear markets as it follows price expansion.
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position size.
# Uses 1d Camarilla levels (R4/S4) calculated from prior day OHLC, aligned to 6b bars.
# Volume filter requires current volume > 1.5x 2-period average to avoid false breakouts.
# Exits when price re-enters prior day's range (mean reversion within the day's bounds).

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_camarilla_breakout_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla levels
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 10:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (using prior day's OHLC)
    pp = np.full(len(df_d), np.nan)
    r4 = np.full(len(df_d), np.nan)
    s4 = np.full(len(df_d), np.nan)
    prev_high = np.full(len(df_d), np.nan)
    prev_low = np.full(len(df_d), np.nan)
    for i in range(1, len(df_d)):
        ph = df_d['high'].iloc[i-1]
        pl = df_d['low'].iloc[i-1]
        pc = df_d['close'].iloc[i-1]
        pp[i] = (ph + pl + pc) / 3.0
        r4[i] = pc + (ph - pl) * 1.1 / 2
        s4[i] = pc - (ph - pl) * 1.1 / 2
        prev_high[i] = ph
        prev_low[i] = pl
    
    # Align daily values to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_d, pp)
    r4_aligned = align_htf_to_ltf(prices, df_d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_d, s4)
    prev_high_aligned = align_htf_to_ltf(prices, df_d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_d, prev_low)
    
    # Volume confirmation: 2-period average (2*6h = 12h ~ half day)
    vol_ma_2 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 2:
            vol_sum -= volume[i-2]
        if i >= 1:
            vol_ma_2[i] = vol_sum / 2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(prev_high_aligned[i]) or 
            np.isnan(prev_low_aligned[i]) or 
            np.isnan(vol_ma_2[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes back inside previous day's range
            if close[i] <= prev_high_aligned[i] and close[i] >= prev_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes back inside previous day's range
            if close[i] <= prev_high_aligned[i] and close[i] >= prev_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above R4 with volume confirmation
            if (close[i] > r4_aligned[i] and 
                volume[i] > vol_ma_2[i] * 1.5):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below S4 with volume confirmation
            elif (close[i] < s4_aligned[i] and 
                  volume[i] > vol_ma_2[i] * 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-09 09:51
