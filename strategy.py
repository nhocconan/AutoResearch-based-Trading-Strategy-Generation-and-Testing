#!/usr/bin/env python3
name = "12h_Donchian20_Breakout_1dTrend_Volume_Confirm"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1D data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1D EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Donchian channels on 12H
    donchian_high = np.zeros(n)
    donchian_low = np.zeros(n)
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.max(high[:i+1])
            donchian_low[i] = np.min(low[:i+1])
        else:
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
    
    # Volume filter: volume > 1.5x 30-period average
    vol_ma30 = np.zeros(n)
    for i in range(n):
        if i < 30:
            vol_ma30[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma30[i] = np.mean(volume[i-29:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 30)
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high, close above 1D EMA34, volume surge
            if (close[i] > donchian_high[i] and 
                close[i] > ema34_1d_aligned[i] and
                volume[i] > 1.5 * vol_ma30[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low, close below 1D EMA34, volume surge
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema34_1d_aligned[i] and
                  volume[i] > 1.5 * vol_ma30[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below Donchian low OR close below 1D EMA34
            if (close[i] < donchian_low[i] or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above Donchian high OR close above 1D EMA34
            if (close[i] > donchian_high[i] or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals