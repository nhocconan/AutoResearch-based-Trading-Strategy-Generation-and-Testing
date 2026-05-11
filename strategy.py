#!/usr/bin/env python3
"""
12h_MidPoint_Reversal_1dTrend_Filter
Hypothesis: Uses price crossing above/below the previous day's midpoint (pivot) as entry signal, filtered by daily trend (1d EMA34) and volume confirmation. Exits when price reverts to the previous day's midpoint or trend reverses. Designed for low trade frequency (12-37/year) with clear signals in both bull and bear markets by following the higher timeframe trend.
"""

name = "12h_MidPoint_Reversal_1dTrend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 12h price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Previous day's high and low (for midpoint calculation)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    # First day has no previous - set to current values to avoid false signals
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Previous day's midpoint (pivot)
    prev_mid = (prev_high + prev_low) / 2.0
    
    # Daily trend filter (1d EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(prev_mid[i]) or np.isnan(ema_34_12h[i]) or 
            np.isnan(vol_ratio[i])):
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
            # Long: price crosses above previous day's midpoint + above daily EMA34 + volume
            if (close[i] > prev_mid[i] and 
                close[i] > ema_34_12h[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below previous day's midpoint + below daily EMA34 + volume
            elif (close[i] < prev_mid[i] and 
                  close[i] < ema_34_12h[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price returns to previous day's midpoint OR trend turns down
                if (close[i] <= prev_mid[i]) or \
                   (close[i] < ema_34_12h[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to previous day's midpoint OR trend turns up
                if (close[i] >= prev_mid[i]) or \
                   (close[i] > ema_34_12h[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals