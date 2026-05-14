# Strategy: 6h_ElderRay_RayPower_1dTrend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.321 | +40.2% | -13.3% | 260 | PASS |
| ETHUSDT | 0.393 | +51.2% | -15.2% | 278 | PASS |
| SOLUSDT | 0.934 | +192.5% | -26.9% | 321 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.313 | +1.3% | -6.8% | 94 | FAIL |
| ETHUSDT | 0.262 | +10.2% | -9.6% | 92 | PASS |
| SOLUSDT | 0.242 | +10.0% | -11.8% | 89 | PASS |

## Code
```python
#!/usr/bin/env python3
name = "6h_ElderRay_RayPower_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Elder Ray components on 1d
    ema13_1d = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = df_1d['high'].values - ema13_1d
    bear_power_1d = df_1d['low'].values - ema13_1d
    
    # Align Elder Ray components to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # 1d trend: EMA 34
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema13_aligned[i]) or np.isnan(ema34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 AND price > EMA13 AND price > EMA34
            if (bull_power_aligned[i] > 0 and 
                close[i] > ema13_aligned[i] and 
                close[i] > ema34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND price < EMA13 AND price < EMA34
            elif (bear_power_aligned[i] < 0 and 
                  close[i] < ema13_aligned[i] and 
                  close[i] < ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 OR price < EMA13 OR price < EMA34
            if (bull_power_aligned[i] <= 0 or 
                close[i] < ema13_aligned[i] or 
                close[i] < ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: Bear Power >= 0 OR price > EMA13 OR price > EMA34
            if (bear_power_aligned[i] >= 0 or 
                close[i] > ema13_aligned[i] or 
                close[i] > ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals
```

## Last Updated
2026-05-11 15:08
