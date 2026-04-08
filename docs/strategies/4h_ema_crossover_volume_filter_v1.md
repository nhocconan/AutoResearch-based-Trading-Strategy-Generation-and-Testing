# Strategy: 4h_ema_crossover_volume_filter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.341 | -4.2% | -21.0% | 134 | FAIL |
| ETHUSDT | -0.159 | +1.2% | -23.9% | 128 | FAIL |
| SOLUSDT | 0.871 | +182.2% | -30.3% | 127 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.414 | +14.6% | -8.9% | 37 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_ema_crossover_volume_filter_v1
Hypothesis: EMA crossover (21/55) with volume confirmation on 4h timeframe. 
Golden cross (EMA21 > EMA55) signals uptrend, death cross signals downtrend. 
Volume filter (current volume > 1.5x average) reduces false signals. 
Works in both bull and bear markets by following trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ema_crossover_volume_filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA indicators
    ema21 = pd.Series(close).ewm(span=21, adjust=False).mean().values
    ema55 = pd.Series(close).ewm(span=55, adjust=False).mean().values
    
    # Volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(55, n):
        # Skip if required data not available
        if np.isnan(ema21[i]) or np.isnan(ema55[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: death cross (EMA21 crosses below EMA55)
            if ema21[i] < ema55[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: golden cross (EMA21 crosses above EMA55)
            if ema21[i] > ema55[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Golden cross long
            if (ema21[i] > ema55[i] and 
                vol_confirm):
                position = 1
                signals[i] = 0.25
            # Death cross short
            elif (ema21[i] < ema55[i] and 
                  vol_confirm):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 15:02
