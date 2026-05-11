#!/usr/bin/env python3
"""
4h_Post_Close_Reversal_With_Trend
Hypothesis: On 4h chart, enter long when price closes above the prior day's high while above weekly EMA50 (bullish reversal), and short when price closes below prior day's low while below weekly EMA50 (bearish reversal). Uses volume confirmation and exits when price reverts to prior day's close or trend weakens. Designed for low trade frequency (<30/year) and works in both bull and bear markets by following weekly trend.
"""

name = "4h_Post_Close_Reversal_With_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prrices)
    if n < 50:
        return np.zeros(n)
    
    # 4h price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily high/low for reference (prior day's session)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Align daily levels to 4h
    daily_high_4h = align_htf_to_ltf(prices, df_1d, daily_high)
    daily_low_4h = align_htf_to_ltf(prices, df_1d, daily_low)
    daily_close_4h = align_htf_to_ltf(prices, df_1d, daily_close)
    
    # Weekly trend filter (1w EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation (20-period average on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(daily_high_4h[i]) or np.isnan(daily_low_4h[i]) or 
            np.isnan(daily_close_4h[i]) or np.isnan(ema_50_4h[i]) or 
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
            # Long: close above prior day's high + above weekly EMA50 + volume spike
            if (close[i] > daily_high_4h[i] and 
                close[i] > ema_50_4h[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: close below prior day's low + below weekly EMA50 + volume spike
            elif (close[i] < daily_low_4h[i] and 
                  close[i] < ema_50_4h[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price returns to prior day's close OR trend turns down
                if (close[i] <= daily_close_4h[i]) or \
                   (close[i] < ema_50_4h[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to prior day's close OR trend turns up
                if (close[i] >= daily_close_4h[i]) or \
                   (close[i] > ema_50_4h[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals