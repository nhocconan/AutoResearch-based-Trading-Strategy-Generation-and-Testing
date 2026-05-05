#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume spike + 1d chop regime filter (CHOP > 61.8 = range)
# Long when: price breaks above 20-period Donchian high AND 1d volume > 1.5x 20-period MA AND 1d CHOP > 61.8
# Short when: price breaks below 20-period Donchian low AND 1d volume > 1.5x 20-period MA AND 1d CHOP > 61.8
# Exit when: price returns to 10-period Donchian midpoint OR volume drops below average
# Uses Donchian for structure, volume for conviction, chop for ranging markets (mean reversion in chop)
# Timeframe: 12h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_Donchian20_1dVolumeChop"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels on 12h (20-period)
    if len(high) >= 20 and len(low) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_high + donchian_low) / 2.0
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Donchian breakout signals
    donchian_break_above = (close > donchian_high) & (np.roll(close, 1) <= np.roll(donchian_high, 1))
    donchian_break_below = (close < donchian_low) & (np.roll(close, 1) >= np.roll(donchian_low, 1))
    donchian_reenter = (close > donchian_mid) & (np.roll(close, 1) <= np.roll(donchian_mid, 1)) | \
                       (close < donchian_mid) & (np.roll(close, 1) >= np.roll(donchian_mid, 1))
    
    # Get 1d data ONCE before loop for volume and chop calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume filter: volume > 1.5x 20-period MA
    if len(volume_1d) >= 20:
        vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume_1d > (1.5 * vol_ma_20)
    else:
        volume_filter = np.full(len(volume_1d), False)
    
    # Calculate 1d Chopiness Index (CHOP) - 14 period
    # CHOP = 100 * log10(sum(ATR(1)) / (max(high,n) - min(low,n))) / log10(n)
    if len(high_1d) >= 14 and len(low_1d) >= 14 and len(close_1d) >= 14:
        # True Range
        tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
        tr1 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
        tr1 = np.concatenate([[np.nan], tr1])  # align with index
        
        # Sum of TR over 14 periods
        sum_tr = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
        
        # Max high and min low over 14 periods
        max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        
        # Avoid division by zero
        denominator = max_high - min_low
        chop_raw = np.where(denominator != 0, 
                           100 * np.log10(sum_tr / denominator) / np.log10(14),
                           50)  # neutral when no range
        
        # CHOP > 61.8 = ranging market (good for mean reversion)
        chop_filter = chop_raw > 61.8
    else:
        chop_filter = np.full(len(high_1d), False)
    
    # Align 1d indicators to 12h timeframe
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter.astype(float))
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_filter.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(volume_filter_aligned[i]) or np.isnan(chop_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Donchian breakout above + 1d volume spike + 1d chop regime (ranging)
            if (donchian_break_above[i] and 
                volume_filter_aligned[i] == 1.0 and 
                chop_filter_aligned[i] == 1.0):
                signals[i] = 0.25
                position = 1
            # Short conditions: Donchian breakout below + 1d volume spike + 1d chop regime (ranging)
            elif (donchian_break_below[i] and 
                  volume_filter_aligned[i] == 1.0 and 
                  chop_filter_aligned[i] == 1.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Donchian channel OR volume drops OR chop breaks down (trending)
            if (donchian_reenter[i] or 
                volume_filter_aligned[i] == 0.0 or 
                chop_filter_aligned[i] == 0.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Donchian channel OR volume drops OR chop breaks down (trending)
            if (donchian_reenter[i] or 
                volume_filter_aligned[i] == 0.0 or 
                chop_filter_aligned[i] == 0.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals