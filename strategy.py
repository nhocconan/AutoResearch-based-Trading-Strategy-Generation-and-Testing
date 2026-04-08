#!/usr/bin/env python3
# 1d_weekly_donchian_breakout_volume_chop_v1
# Hypothesis: Daily Donchian channel breakouts with volume confirmation and weekly choppiness regime filter.
# Long: Price breaks above 20-day Donchian upper band with volume > 1.8x 20-day average AND weekly chop > 61.8 (ranging/mean-reversion favorable for breakouts)
# Short: Price breaks below 20-day Donchian lower band with volume > 1.8x 20-day average AND weekly chop > 61.8
# Exit: Price returns to 20-day Donchian midpoint OR opposite band touch with volume confirmation
# Uses 1d primary timeframe with 1w HTF for choppiness regime filter.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag and avoid overtrading.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_donchian_breakout_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 20-day Donchian channels
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    donchian_mid = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2.0
    
    # Calculate volume ratio (current vs 20-day average)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    vol_ratio = np.where(vol_sma > 0, volume / vol_sma, 0)
    
    # Get 1w data for choppiness regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly choppiness index (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    atr_1w = np.zeros(len(df_1w))
    tr_1w = np.zeros(len(df_1w))
    for i in range(1, len(df_1w)):
        tr_1w[i] = max(high_1w[i] - low_1w[i], 
                       abs(high_1w[i] - close_1w[i-1]), 
                       abs(low_1w[i] - close_1w[i-1]))
    
    for i in range(1, len(df_1w)):
        atr_1w[i] = 0.9 * atr_1w[i-1] + 0.1 * tr_1w[i] if i > 1 else tr_1w[i]
    
    chop_1w = np.full(len(df_1w), np.nan)
    for i in range(14, len(df_1w)):
        sum_atr = np.sum(atr_1w[i-13:i+1])
        highest_high = np.max(high_1w[i-13:i+1])
        lowest_low = np.min(low_1w[i-13:i+1])
        if highest_high > lowest_low:
            chop_1w[i] = 100 * np.log10(sum_atr / (highest_high - lowest_low)) / np.log10(14)
        else:
            chop_1w[i] = 50.0  # neutral when no range
    
    # Align 1d indicators (already aligned as we're in 1d timeframe)
    # Align 1w choppiness to 1d timeframe
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        vol_r = vol_ratio[i]
        price = close[i]
        
        if np.isnan(vol_r):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        dh = donchian_high[i]
        dl = donchian_low[i]
        dm = donchian_mid[i]
        chop = chop_1w_aligned[i]
        
        if np.isnan(dh) or np.isnan(dl) or np.isnan(dm) or np.isnan(chop):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to midpoint or touches lower band with volume
            if price <= dm or (price < dl and vol_r > 1.5):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to midpoint or touches upper band with volume
            if price >= dm or (price > dh and vol_r > 1.5):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: price breaks above upper band with volume AND weekly chop > 61.8 (ranging market)
            if price > dh and vol_r > 1.8 and chop > 61.8:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower band with volume AND weekly chop > 61.8 (ranging market)
            elif price < dl and vol_r > 1.8 and chop > 61.8:
                position = -1
                signals[i] = -0.25
    
    return signals