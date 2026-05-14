# Strategy: 4h_Donchian20_12hEMA50_VolumeConfirm_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.192 | +10.9% | -13.7% | 263 | FAIL |
| ETHUSDT | 0.003 | +19.1% | -11.5% | 236 | PASS |
| SOLUSDT | 0.520 | +73.3% | -23.5% | 203 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.007 | +5.6% | -7.2% | 85 | PASS |
| SOLUSDT | -0.986 | -10.5% | -17.5% | 70 | FAIL |

## Code
```python
#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Uses 12h EMA50 for trend alignment, 4h Donchian(20) for breakout entry, and volume spike (>1.8x 20-bar avg) for confirmation.
# Designed for moderate trade frequency (target 75-200 total over 4 years) to balance edge capture and fee drag.
# Works in both bull and bear markets by following the 12h trend direction and requiring volume confirmation to avoid false breakouts.

name = "4h_Donchian20_12hEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels from prior 20 bars (primary TF)
    lookback = 20
    # Upper = max(high) of last 20 bars, Lower = min(low) of last 20 bars
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    
    # Calculate average volume for confirmation (20-period LTF)
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback, 1), n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(upper[i]) or 
            np.isnan(lower[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper, close > 12h EMA50, volume spike (>1.8x avg)
            if (high[i] > upper[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower, close < 12h EMA50, volume spike (>1.8x avg)
            elif (low[i] < lower[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close position if price breaks below Donchian lower or volume drops
            if (low[i] < lower[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close position if price breaks above Donchian upper or volume drops
            if (high[i] > upper[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-13 22:48
