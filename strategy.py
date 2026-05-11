#!/usr/bin/env python3
"""
1d_HighLowBreakout_WeeklyTrend_Filter
Hypothesis: Uses daily price breaking above yesterday's high or below yesterday's low as entry signal, filtered by weekly trend (1w EMA50) and volume confirmation. Exits when price reverts to the previous day's close or trend reverses. Designed for low trade frequency (<25/year) with clear signals in both bull and bear markets by following the higher timeframe trend.
"""

name = "1d_HighLowBreakout_WeeklyTrend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Daily price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Previous day's high and low (for breakout detection)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # First day has no previous - set to current values to avoid false signals
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Weekly trend filter (1w EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_1d = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation (20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or 
            np.isnan(ema_50_1d[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: price breaks above previous day's high + above weekly EMA50 + volume
            if (close[i] > prev_high[i] and 
                close[i] > ema_50_1d[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below previous day's low + below weekly EMA50 + volume
            elif (close[i] < prev_low[i] and 
                  close[i] < ema_50_1d[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price returns to previous day's close OR trend turns down
                if (close[i] <= prev_close[i]) or \
                   (close[i] < ema_50_1d[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to previous day's close OR trend turns up
                if (close[i] >= prev_close[i]) or \
                   (close[i] > ema_50_1d[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals