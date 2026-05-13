#!/usr/bin/env python3
"""
4H_Price_Channel_Breakout_Volume_Filter
Hypothesis: Price channel breakouts (using Donchian channels) capture strong directional moves, while volume surge confirms institutional participation. The 12h trend filter ensures alignment with higher timeframe momentum, reducing false signals. Works in bull markets by catching breakouts and in bear markets by capturing sharp reversals with volume confirmation. Uses tight entry conditions to limit trade frequency and avoid fee drag.
"""

name = "4H_Price_Channel_Breakout_Volume_Filter"
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
    
    # Donchian channel (20-period)
    dc_period = 20
    upper_channel = np.full_like(close, np.nan)
    lower_channel = np.full_like(close, np.nan)
    
    for i in range(dc_period - 1, len(high)):
        upper_channel[i] = np.max(high[i - dc_period + 1:i + 1])
        lower_channel[i] = np.min(low[i - dc_period + 1:i + 1])
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 50-period EMA on 12h for trend filter
    ema_period = 50
    ema_12h = np.zeros_like(close_12h)
    ema_12h[:] = np.nan
    if len(close_12h) >= ema_period:
        ema_12h[ema_period - 1] = np.mean(close_12h[:ema_period])
        for i in range(ema_period, len(close_12h)):
            ema_12h[i] = (close_12h[i] * (2 / (ema_period + 1))) + \
                         (ema_12h[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Align 12h EMA to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume average (20-period) for volume spike filter
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 2.0x 20-period average
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above upper Donchian + volume spike + price above 12h EMA
            if (close[i] > upper_channel[i] and vol_spike and 
                close[i] > ema_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian + volume spike + price below 12h EMA
            elif (close[i] < lower_channel[i] and vol_spike and 
                  close[i] < ema_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below lower Donchian or loss of volume spike
            if (close[i] < lower_channel[i] or not vol_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above upper Donchian or loss of volume spike
            if (close[i] > upper_channel[i] or not vol_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals