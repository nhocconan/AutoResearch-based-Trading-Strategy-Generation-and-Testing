# Strategy: 4h_Donchian20_12hEMA50_Trend_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.042 | +19.7% | -8.8% | 358 | FAIL |
| ETHUSDT | 0.082 | +23.8% | -7.1% | 323 | PASS |
| SOLUSDT | 0.294 | +38.9% | -18.6% | 295 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.172 | +7.7% | -6.7% | 123 | PASS |
| SOLUSDT | 0.258 | +8.6% | -7.2% | 107 | PASS |

## Code
```python
#!/usr/bin/env python3
# Hypothesis: 4h Donchian channel breakout with 12h EMA trend filter and volume confirmation.
# Long when price breaks above Donchian upper (20) and close > 12h EMA50 with volume > 1.8x 20-bar average.
# Short when price breaks below Donchian lower (20) and close < 12h EMA50 with volume > 1.8x 20-bar average.
# Uses discrete sizing 0.25 to target 75-200 total trades over 4 years on 4h timeframe.
# Donchian channels provide clear structure; 12h EMA filters counter-trend noise; volume confirms momentum.
# Designed for fewer, higher-quality trades to avoid fee drag while working in both bull and bear markets.

name = "4h_Donchian20_12hEMA50_Trend_VolumeConfirm"
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
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period) from prior candles only
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper, close > 12h EMA50, volume spike
            if (high[i] > highest_high[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower, close < 12h EMA50, volume spike
            elif (low[i] < lowest_low[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian lower OR volume drops below average
            if (low[i] < lowest_low[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian upper OR volume drops below average
            if (high[i] > highest_high[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-13 22:04
