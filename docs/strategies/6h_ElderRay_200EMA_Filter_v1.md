# Strategy: 6h_ElderRay_200EMA_Filter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.211 | +5.5% | -14.9% | 554 | FAIL |
| ETHUSDT | 0.200 | +32.0% | -16.8% | 567 | PASS |
| SOLUSDT | 0.953 | +191.2% | -26.1% | 557 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.111 | +6.8% | -11.9% | 192 | PASS |
| SOLUSDT | 0.215 | +9.2% | -11.6% | 165 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
6h_ElderRay_200EMA_Filter_v1
Elder Ray (Bull/Bear Power) + 200-period EMA trend filter on 6h timeframe.
Bull Power = High - EMA13, Bear Power = EMA13 - Low.
Long when Bull Power > 0 and price > EMA200 (uptrend).
Short when Bear Power > 0 and price < EMA200 (downtrend).
Exit when power reverses or price crosses EMA200.
Designed to capture momentum in trending markets while avoiding counter-trend trades.
Target: 50-150 total trades over 4 years (12-37/year).
"""

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
    
    # === 60-period EMA for Elder Ray (13-period equivalent scaled to 6h) ===
    # Using 60 to approximate 13 periods on higher timeframe for smoother signal
    alpha = 2 / (60 + 1)
    ema60 = np.zeros_like(close)
    ema60[0] = close[0]
    for i in range(1, n):
        ema60[i] = ema60[i-1] + alpha * (close[i] - ema60[i-1])
    
    # === 200-period EMA for trend filter ===
    alpha200 = 2 / (200 + 1)
    ema200 = np.zeros_like(close)
    ema200[0] = close[0]
    for i in range(1, n):
        ema200[i] = ema200[i-1] + alpha200 * (close[i] - ema200[i-1])
    
    # === Elder Ray components ===
    bull_power = high - ema60  # High - EMA13 equivalent
    bear_power = ema60 - low   # EMA13 - Low equivalent
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema60[i]) or 
            np.isnan(ema200[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Bull Power positive AND price above EMA200 (uptrend)
            if (bull_power[i] > 0 and 
                close[i] > ema200[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Bear Power positive AND price below EMA200 (downtrend)
            elif (bear_power[i] > 0 and 
                  close[i] < ema200[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: Bear Power becomes positive OR price crosses below EMA200
            if (bear_power[i] > 0 or 
                close[i] < ema200[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bull Power becomes positive OR price crosses above EMA200
            if (bull_power[i] > 0 or 
                close[i] > ema200[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_200EMA_Filter_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-17 02:44
