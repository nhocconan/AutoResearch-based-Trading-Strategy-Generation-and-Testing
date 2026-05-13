#!/usr/bin/env python3
"""
1D_Donchian_Breakout_With_Volume_Filter
Hypothesis: Daily Donchian channel breakouts capture major trend moves, while volume confirmation filters false breakouts. Works in bull markets by catching sustained uptrends and in bear markets by capturing sharp reversals with volume spikes. Uses weekly trend filter to avoid counter-trend trades.
"""

name = "1D_Donchian_Breakout_With_Volume_Filter"
timeframe = "1d"
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
    
    # Calculate Donchian channels (20-day)
    donchian_period = 20
    upper_channel = np.full_like(high, np.nan)
    lower_channel = np.full_like(low, np.nan)
    
    for i in range(donchian_period - 1, n):
        upper_channel[i] = np.max(high[i-donchian_period+1:i+1])
        lower_channel[i] = np.min(low[i-donchian_period+1:i+1])
    
    # Get weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly 50-period EMA for trend filter
    ema_period = 50
    ema_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= ema_period:
        multiplier = 2 / (ema_period + 1)
        ema_1w[ema_period-1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * multiplier) + (ema_1w[i-1] * (1 - multiplier))
    
    # Align weekly EMA to daily
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume average (20-day) for volume confirmation
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-day average
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above upper Donchian + volume spike + weekly uptrend
            if (close[i] > upper_channel[i] and vol_spike and 
                close[i] > ema_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian + volume spike + weekly downtrend
            elif (close[i] < lower_channel[i] and vol_spike and 
                  close[i] < ema_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below lower Donchian or weekly trend turns down
            if (close[i] < lower_channel[i] or close[i] < ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above upper Donchian or weekly trend turns up
            if (close[i] > upper_channel[i] or close[i] > ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals