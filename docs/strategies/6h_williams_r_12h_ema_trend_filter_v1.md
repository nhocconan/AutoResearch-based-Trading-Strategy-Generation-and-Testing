# Strategy: 6h_williams_r_12h_ema_trend_filter_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.030 | +19.6% | -10.1% | 145 | FAIL |
| ETHUSDT | -0.769 | -13.6% | -30.7% | 149 | FAIL |
| SOLUSDT | 0.206 | +32.1% | -23.2% | 137 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.219 | +8.7% | -10.4% | 57 | PASS |

## Code
```python
# -*- coding: utf-8 -*-
# -*- mode: python; -*-

#!/usr/bin/env python3
"""
6h Williams %R + 12h EMA Trend Filter
Long when Williams %R < -80 (oversold) and price above 12h EMA50
Short when Williams %R > -20 (overbought) and price below 12h EMA50
Exit when Williams %R crosses -50 (mean reversion)
Designed for mean-reversion in ranging markets with trend filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_williams_r_12h_ema_trend_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === Williams %R (14) ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    willr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # === 12h EMA50 Trend Filter ===
    df_12h = get_htf_data(prices, '12h')
    ema_50 = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        if np.isnan(willr[i]) or np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R crosses above -50 (overbought territory)
            if willr[i] > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -50 (oversold territory)
            if willr[i] < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Mean reentry conditions with trend filter
            if willr[i] < -80 and close[i] > ema_50_aligned[i]:  # Oversold + above trend
                position = 1
                signals[i] = 0.25
            elif willr[i] > -20 and close[i] < ema_50_aligned[i]:  # Overbought + below trend
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 23:27
