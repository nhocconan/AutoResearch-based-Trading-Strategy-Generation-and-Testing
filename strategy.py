#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ChoppinessIndex_Breakout_Direction_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Choppiness Index (regime filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate daily Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    atr_1d = np.zeros(len(df_1d))
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], 
                       np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                                  np.abs(low_1d[1:] - close_1d[:-1])))
    tr_1d = np.concatenate([[np.inf], tr_1d])  # First TR is undefined
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(ATR14) / (max(HH14) - min(LL14))) / log10(14)
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    range_14 = highest_high_1d - lowest_low_1d
    chop = 100 * np.log10(atr_sum / range_14) / np.log10(14)
    chop = np.where(range_14 == 0, 100, chop)  # Avoid division by zero
    
    chop_align = align_htf_to_ltf(prices, df_1d, chop)
    
    # Get 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.3 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Need enough data for Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(chop_align[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop_align[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: break above upper channel in trending market (Chop < 38.2)
            if close[i] > upper_channel and chop_val < 38.2 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: break below lower channel in trending market (Chop < 38.2)
            elif close[i] < lower_channel and chop_val < 38.2 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below lower channel (mean reversion signal)
            if close[i] < lower_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above upper channel (mean reversion signal)
            if close[i] > upper_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals