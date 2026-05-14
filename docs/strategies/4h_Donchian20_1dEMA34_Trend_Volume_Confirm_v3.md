# Strategy: 4h_Donchian20_1dEMA34_Trend_Volume_Confirm_v3

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.248 | +30.9% | -11.6% | 135 | PASS |
| ETHUSDT | 0.564 | +52.4% | -12.9% | 124 | PASS |
| SOLUSDT | 0.628 | +79.0% | -21.0% | 119 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.136 | -3.3% | -5.1% | 55 | FAIL |
| ETHUSDT | 0.731 | +16.7% | -5.5% | 45 | PASS |
| SOLUSDT | -0.011 | +5.3% | -10.4% | 38 | FAIL |

## Code
```python
#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA(34) trend filter and volume confirmation.
# Long when price breaks above Donchian upper band with price > 1d EMA34 (bullish trend) and volume > 2.0x 20-bar average.
# Short when price breaks below Donchian lower band with price < 1d EMA34 (bearish trend) and volume > 2.0x average.
# Exit when price reverses and closes below/above the midpoint of the Donchian channel.
# Uses discrete position sizing 0.25. Target: 75-200 total trades over 4 years on 4h timeframe.
# EMA trend filter ensures we trade with the higher timeframe trend, avoiding counter-trend whipsaws.
# Volume confirmation validates breakout strength. Donchian midpoint exit provides clear, objective stop.

name = "4h_Donchian20_1dEMA34_Trend_Volume_Confirm_v3"
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
    
    # Calculate Donchian channel (20-period)
    lookback = 20
    if n < lookback + 1:
        return np.zeros(n)
    
    upper_band = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    lower_band = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    mid_band = (upper_band + lower_band) / 2
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close
    if len(close_1d) < 34:
        ema_34_1d = np.full(len(close_1d), np.nan)
    else:
        ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 4h timeframe (wait for 1d bar to close)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback + 20, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper band with bullish 1d EMA trend and volume spike
            if (close[i] > upper_band[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower band with bearish 1d EMA trend and volume spike
            elif (close[i] < lower_band[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below mid-band (reversal signal)
            if close[i] < mid_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above mid-band (reversal signal)
            if close[i] > mid_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-13 21:29
