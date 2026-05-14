# Strategy: 6h_ElderRay_12hEMA20_Trend_VolumeConfirm

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.676 | -1.0% | -15.2% | 508 | FAIL |
| ETHUSDT | 0.121 | +25.5% | -8.2% | 467 | PASS |
| SOLUSDT | 0.793 | +91.7% | -15.8% | 365 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.575 | +13.5% | -7.2% | 155 | PASS |
| SOLUSDT | -1.723 | -14.9% | -18.7% | 131 | FAIL |

## Code
```python
#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray Index with 12h EMA trend filter and volume confirmation.
# Elder Ray measures bull/bear power relative to EMA13. Long when bull power > 0 and bear power < 0 with expanding volume.
# Short when bear power < 0 and bull power > 0 with expanding volume. Uses 12h EMA20 for trend filter.
# Volume must be > 1.3x 20-bar average for confirmation. Exits when power diverges or volume dries up.
# Designed to work in both bull (strong bull power) and bear (strong bear power) markets by measuring actual buying/selling pressure.
# Discrete sizing 0.25 targets 50-150 total trades over 4 years on 6h timeframe.

name = "6h_ElderRay_12hEMA20_Trend_VolumeConfirm"
timeframe = "6h"
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
    
    lookback = 20  # for EMA and volume average
    
    # Calculate EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # Bull power: high minus EMA13
    bear_power = low - ema13   # Bear power: low minus EMA13
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA(20) on 12h close
    if len(close_12h) < 20:
        ema_20_12h = np.full(len(close_12h), np.nan)
    else:
        ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 12h EMA to 6h timeframe (wait for 12h bar to close)
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_20_12h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull power positive AND bear power negative (bulls in control) with volume spike and bullish 12h trend
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                close[i] > ema_20_12h_aligned[i] and 
                volume[i] > 1.3 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear power negative AND bull power positive (bears in control) with volume spike and bearish 12h trend
            elif (bear_power[i] < 0 and bull_power[i] > 0 and 
                  close[i] < ema_20_12h_aligned[i] and 
                  volume[i] > 1.3 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull power turns negative OR volume dries up (< 0.8x average)
            if bull_power[i] <= 0 or volume[i] < 0.8 * avg_volume[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear power turns positive OR volume dries up (< 0.8x average)
            if bear_power[i] >= 0 or volume[i] < 0.8 * avg_volume[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-13 21:39
