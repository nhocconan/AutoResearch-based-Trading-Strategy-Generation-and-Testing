# Strategy: 4h_12h_camarilla_ema50_volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.007 | +19.6% | -11.5% | 168 | FAIL |
| ETHUSDT | 0.136 | +26.6% | -14.8% | 154 | PASS |
| SOLUSDT | 0.515 | +64.4% | -29.4% | 142 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.176 | +25.1% | -6.8% | 49 | PASS |
| SOLUSDT | 0.055 | +6.2% | -11.5% | 43 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_12h_camarilla_ema50_volume_v1
Hypothesis: 4-hour strategy using 12-hour EMA50 for trend direction and 12-hour Camarilla pivot levels for entries, with volume confirmation.
Works in bull/bear by requiring alignment with the 12h trend (EMA50) and confirming with volume to avoid false breakouts.
Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend and Camarilla
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend direction
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Previous 12h bar's range for Camarilla
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_close_12h = np.roll(close_12h, 1)
    
    range_12h = prev_high_12h - prev_low_12h
    # Resistance levels
    r3 = prev_close_12h + range_12h * 1.1 / 2
    r4 = prev_close_12h + range_12h * 1.1
    # Support levels
    s3 = prev_close_12h - range_12h * 1.1 / 2
    s4 = prev_close_12h - range_12h * 1.1
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price > EMA50 (uptrend) AND close breaks above R4 with volume
        if (close[i] > ema50_12h_aligned[i] and close[i] > r4_aligned[i] and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price < EMA50 (downtrend) AND close breaks below S4 with volume
        elif (close[i] < ema50_12h_aligned[i] and close[i] < s4_aligned[i] and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or close crosses back to opposite S3/R3
        elif position == 1 and close[i] < s3_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > r3_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_camarilla_ema50_volume_v1"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-12 17:29
