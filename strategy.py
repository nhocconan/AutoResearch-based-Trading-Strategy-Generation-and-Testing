#!/usr/bin/env python3
# 12h_donchian_20_volume_regime_v1
# Hypothesis: 12h Donchian(20) breakouts with volume confirmation and 1d chop regime filter capture trending moves while avoiding false signals in ranging markets. Works in bull/bear by filtering out chop.
# Target: 12-37 trades/year (50-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_20_volume_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20-1, n):
        highest_high[i] = np.max(high[i-20+1:i+1])
        lowest_low[i] = np.min(low[i-20+1:i+1])
    
    # Volume confirmation: 20-period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    # Get 1d data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate chop index (14-period) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr_1d = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        tr_1d[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
    
    atr_1d = np.zeros(len(df_1d))
    atr_1d[0] = tr_1d[0]
    for i in range(1, len(df_1d)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    chop = np.zeros(len(df_1d))
    for i in range(14, len(df_1d)):
        sum_atr = np.sum(atr_1d[i-14:i+1])
        max_high = np.max(high_1d[i-14:i+1])
        min_low = np.min(low_1d[i-14:i+1])
        if max_high > min_low and sum_atr > 0:
            chop[i] = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    
    # Align chop to 12h timeframe (trending when chop < 38.2)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_ma_20[i]) or np.isnan(chop_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma_20[i] * 2.0
        
        # Chop regime filter: trending when chop < 38.2
        trending = chop_aligned[i] < 38.2
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or chop increases (range)
            if close[i] < lowest_low[i] or chop_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or chop increases (range)
            if close[i] > highest_high[i] or chop_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high with volume and trending
            if close[i] > highest_high[i] and vol_spike and trending:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with volume and trending
            elif close[i] < lowest_low[i] and vol_spike and trending:
                position = -1
                signals[i] = -0.25
    
    return signals