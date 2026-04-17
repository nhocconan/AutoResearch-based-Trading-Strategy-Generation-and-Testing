#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation (1.5x 20-period average) 
and trend filter using 1d EMA50. Enter long when price breaks above upper Donchian 
channel with volume spike and price > EMA50; enter short when price breaks below 
lower Donchian channel with volume spike and price < EMA50. Exit when price 
returns to the middle of the channel (EMA20 of Donchian bounds). 
Position size: 0.30 for entries, 0 for exits.
Target: 20-50 total trades over 4 years (5-12.5/year) to stay under 400 total 4h trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian Channels (20)
    upper_dc = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_dc = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    middle_dc = (upper_dc + lower_dc) / 2
    
    # Get 1d data for EMA50 filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA (50)
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 4h
    upper_dc_aligned = align_htf_to_ltf(prices, df_4h, upper_dc)
    lower_dc_aligned = align_htf_to_ltf(prices, df_4h, lower_dc)
    middle_dc_aligned = align_htf_to_ltf(prices, df_4h, middle_dc)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_dc_aligned[i]) or np.isnan(lower_dc_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian, volume spike, price > EMA50
            if (close[i] > upper_dc_aligned[i] and 
                volume[i] > vol_ma_20_aligned[i] * 1.5 and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below lower Donchian, volume spike, price < EMA50
            elif (close[i] < lower_dc_aligned[i] and 
                  volume[i] > vol_ma_20_aligned[i] * 1.5 and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Exit long: price returns to middle of Donchian channel
            if close[i] <= middle_dc_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit short: price returns to middle of Donchian channel
            if close[i] >= middle_dc_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Volume_EMA50"
timeframe = "4h"
leverage = 1.0