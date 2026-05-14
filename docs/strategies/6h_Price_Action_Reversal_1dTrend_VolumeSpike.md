# Strategy: 6h_Price_Action_Reversal_1dTrend_VolumeSpike

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.689 | +50.7% | -5.6% | 173 | PASS |
| ETHUSDT | 0.396 | +38.9% | -7.7% | 148 | PASS |
| SOLUSDT | 0.500 | +61.2% | -20.6% | 126 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.118 | -4.4% | -7.6% | 73 | FAIL |
| ETHUSDT | 0.411 | +11.8% | -7.1% | 52 | PASS |
| SOLUSDT | -0.164 | +2.8% | -14.9% | 50 | FAIL |

## Code
```python
#!/usr/bin/env python3
name = "6h_Price_Action_Reversal_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Data for trend and price action ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 1d EMA34 for trend ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 1d Price Action: Higher High/Lower Low pattern ===
    # Higher High: current high > previous high
    # Lower Low: current low < previous low
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    higher_high = high_1d > prev_high_1d
    lower_low = low_1d < prev_low_1d
    
    # Align to 6h timeframe
    higher_high_aligned = align_htf_to_ltf(prices, df_1d, higher_high.astype(float))
    lower_low_aligned = align_htf_to_ltf(prices, df_1d, lower_low.astype(float))
    
    # === Volume spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(higher_high_aligned[i]) or
            np.isnan(lower_low_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: HH on 1d + volume spike + price above 1d EMA
            if (higher_high_aligned[i] == 1.0 and 
                volume_spike[i] and
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: LL on 1d + volume spike + price below 1d EMA
            elif (lower_low_aligned[i] == 1.0 and 
                  volume_spike[i] and
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: LL on 1d or price below EMA
            if lower_low_aligned[i] == 1.0 or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: HH on 1d or price above EMA
            if higher_high_aligned[i] == 1.0 or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 05:31
