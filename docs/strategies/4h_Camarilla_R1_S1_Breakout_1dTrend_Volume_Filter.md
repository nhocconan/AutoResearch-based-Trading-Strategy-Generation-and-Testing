# Strategy: 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.440 | +50.7% | -12.9% | 100 | PASS |
| ETHUSDT | 0.058 | +19.5% | -20.3% | 110 | PASS |
| SOLUSDT | 0.750 | +142.3% | -32.6% | 122 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.417 | -0.2% | -7.6% | 44 | FAIL |
| ETHUSDT | 0.816 | +24.0% | -9.9% | 32 | PASS |
| SOLUSDT | 0.623 | +20.5% | -10.4% | 33 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Filter
# Hypothesis: Price breaking above/below daily Camarilla R1/S1 levels with 1-day EMA34 trend filter and volume confirmation captures strong trending moves while avoiding false breakouts. Works in bull/bear by following the higher timeframe trend direction. Uses 4h timeframe with 1d EMA34 trend filter for higher timeframe context.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Filter"
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

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # Calculate daily high, low, close for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla levels: R1, S1
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    camarilla_range = high_1d - low_1d
    r1_level = close_1d + 1.1 * camarilla_range / 12
    s1_level = close_1d - 1.1 * camarilla_range / 12

    # Align Camarilla levels to 4h timeframe
    r1_level_aligned = align_htf_to_ltf(prices, df_1d, r1_level)
    s1_level_aligned = align_htf_to_ltf(prices, df_1d, s1_level)

    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):  # Start after EMA34 warmup
        if (np.isnan(r1_level_aligned[i]) or np.isnan(s1_level_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + 1d EMA34 uptrend + volume confirmation
            if (close[i] > r1_level_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + 1d EMA34 downtrend + volume confirmation
            elif (close[i] < s1_level_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 1d EMA34 (trend reversal)
            if close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above 1d EMA34 (trend reversal)
            if close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-12 14:25
