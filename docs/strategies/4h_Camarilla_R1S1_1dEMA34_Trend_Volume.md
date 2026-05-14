# Strategy: 4h_Camarilla_R1S1_1dEMA34_Trend_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.347 | +31.4% | -7.3% | 314 | PASS |
| ETHUSDT | 0.275 | +30.6% | -10.0% | 285 | PASS |
| SOLUSDT | 0.367 | +40.3% | -17.4% | 241 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.664 | -3.2% | -5.4% | 120 | FAIL |
| ETHUSDT | 0.542 | +11.4% | -4.2% | 111 | PASS |
| SOLUSDT | 0.075 | +6.6% | -5.4% | 88 | PASS |

## Code
```python
# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Hypothesis: 4-hour Camarilla R1/S1 breakout with daily EMA trend and volume spike.
This strategy targets mean-reversion breakouts in ranging markets while filtering with
higher-timeframe trend to avoid counter-trend trades. Volume spikes confirm institutional
interest. The combination should work in both bull and bear regimes by adapting to
the daily trend filter. Target: 25-40 trades/year per symbol.
"""

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
    
    # Load daily data - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate daily OHLC for Camarilla pivot levels
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # Previous day's OHLC for today's pivot levels
    high_prev = np.roll(high_d, 1)
    low_prev = np.roll(low_d, 1)
    close_prev = np.roll(close_d, 1)
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    pivot = (high_prev + low_prev + close_prev) / 3.0
    range_val = high_prev - low_prev
    
    # Camarilla levels (S1 and R1)
    s1 = close_prev - (range_val * 1.1 / 12)
    r1 = close_prev + (range_val * 1.1 / 12)
    
    # Align all levels to 4h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(ema34_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price closes above R1 with bullish daily trend and volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema34_aligned[i] and  # Bullish trend: price above EMA34
                volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price closes below S1 with bearish daily trend and volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema34_aligned[i] and  # Bearish trend: price below EMA34
                  volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to pivot point
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to pivot
                if close[i] <= pivot_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to pivot
                if close[i] >= pivot_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1S1_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-22 17:26
