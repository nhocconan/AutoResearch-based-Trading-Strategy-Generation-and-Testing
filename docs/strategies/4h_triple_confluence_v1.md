# Strategy: 4h_triple_confluence_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.354 | -0.2% | -20.7% | 131 | FAIL |
| ETHUSDT | -0.288 | -2.9% | -20.3% | 128 | FAIL |
| SOLUSDT | 0.605 | +98.5% | -26.1% | 130 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.398 | +13.1% | -11.3% | 37 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_triple_confluence_v1
# Hypothesis: Combines 4h Donchian breakout, 12h EMA trend filter, and volume spike confirmation.
# Designed to work in both bull and bear markets by requiring multiple confluence factors.
# Target: 20-40 trades/year (80-160 total over 4 years) with strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_triple_confluence_v1"
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
    
    # 1. 4h Donchian Channel (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    # Calculate rolling max/min with proper handling
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # 2. 12h EMA trend filter (21-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema21_12h = np.zeros(len(df_12h))
    ema21_12h[0] = close_12h[0]
    alpha = 2 / (21 + 1)
    for i in range(1, len(df_12h)):
        ema21_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema21_12h[i-1]
    
    # Trend: 1 if close > EMA21, -1 if close < EMA21
    trend_12h = np.where(close_12h > ema21_12h, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # 3. Volume confirmation (20-period average)
    vol_ma_20 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(trend_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR trend turns bearish
            if close[i] < donchian_low[i] or trend_12h_aligned[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR trend turns bullish
            if close[i] > donchian_high[i] or trend_12h_aligned[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high with volume and bullish trend
            if (close[i] > donchian_high[i] and 
                vol_ok and 
                trend_12h_aligned[i] == 1):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with volume and bearish trend
            elif (close[i] < donchian_low[i] and 
                  vol_ok and 
                  trend_12h_aligned[i] == -1):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-09 06:47
