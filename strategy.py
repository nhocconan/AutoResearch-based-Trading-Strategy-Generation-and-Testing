#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Choppiness_Filtered_Donchian_With_Exit"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 120:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for choppiness index (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period Choppiness Index on daily data
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14 = highest_high_14 - lowest_low_14
    range_14[range_14 == 0] = 1e-10
    
    chop = 100 * np.log10(atr14 / range_14) / np.log10(14)
    chop = np.where(np.isnan(chop), 50.0, chop)  # fill NaN with neutral value
    
    # Align choppiness to 4h timeframe
    chop_4h = align_htf_to_ltf(prices, df_1d, chop)
    
    # 4h Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 120
    
    for i in range(start_idx, n):
        if np.isnan(chop_4h[i]) or np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        chop_val = chop_4h[i]
        upper_channel = highest_high_20[i]
        lower_channel = lowest_low_20[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        # Chop > 61.8 indicates ranging market (good for mean reversion)
        ranging_market = chop_val > 61.8
        
        if position == 0:
            # Long: Price breaks below lower channel in ranging market + volume
            if price < lower_channel and ranging_market and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks above upper channel in ranging market + volume
            elif price > upper_channel and ranging_market and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns to middle of channel or breaks upper channel
            middle = (upper_channel + lower_channel) / 2
            if price > middle or price > upper_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns to middle of channel or breaks lower channel
            middle = (upper_channel + lower_channel) / 2
            if price < middle or price < lower_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals