# Strategy: 6h_ElderRay_1dTrend_Volume_Signal

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.105 | +24.7% | -14.9% | 76 | PASS |
| ETHUSDT | 0.084 | +22.3% | -16.2% | 75 | PASS |
| SOLUSDT | 0.749 | +130.7% | -34.7% | 75 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.397 | +0.3% | -7.0% | 27 | FAIL |
| ETHUSDT | 0.343 | +12.0% | -8.5% | 29 | PASS |
| SOLUSDT | 0.334 | +12.3% | -10.7% | 21 | PASS |

## Code
```python
#!/usr/bin/env python3
# 6h_ElderRay_1dTrend_Volume_Signal
# Hypothesis: Elder Ray (Bull/Bear Power) captures bull/bear strength via EMA13 deviation. 
# Combined with 1-day EMA34 trend filter and volume confirmation to enter in direction of higher timeframe trend.
# Works in bull markets (buy on bullish power + uptrend) and bear markets (sell on bearish power + downtrend).
# Volume spike filters low-conviction moves. Target: 20-40 trades/year per symbol.

name = "6h_ElderRay_1dTrend_Volume_Signal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Elder Ray and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 for Elder Ray (using close)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Align Bull/Bear Power to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # 1-day EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike detection: 2.0x average volume (48-period = 2 days on 6h chart)
    vol_ma = pd.Series(volume).rolling(window=48, min_periods=48).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 48)  # Ensure we have EMA34 and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (strength), price above EMA34 (uptrend), volume spike
            if (bull_power_aligned[i] > 0 and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (weakness), price below EMA34 (downtrend), volume spike
            elif (bear_power_aligned[i] < 0 and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bull Power turns negative (loss of strength) OR price crosses below EMA34
            if (bull_power_aligned[i] <= 0 or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bear Power turns positive (loss of weakness) OR price crosses above EMA34
            if (bear_power_aligned[i] >= 0 or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-07 02:25
