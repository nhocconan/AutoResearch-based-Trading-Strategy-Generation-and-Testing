#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot from 1d + volume spike + Choppiness regime
# Long when price touches/breaks above H3 with volume > 2x 20-period average and Choppiness < 61.8 (trending)
# Short when price touches/breaks below L3 with volume > 2x 20-period average and Choppiness < 61.8
# Exit when price crosses the 1d pivot point (central level)
# Uses 1d Camarilla levels as institutional support/resistance, volume for conviction, Choppiness to avoid ranging markets
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla levels and Choppiness
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and ranges
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Camarilla levels: H3, L3, and pivot
    camarilla_h3 = close_1d + (range_hl * 1.1 / 2)
    camarilla_l3 = close_1d - (range_hl * 1.1 / 2)
    
    # Calculate Choppiness Index (14-period)
    atr_1d = []
    tr_1d = []
    for i in range(len(df_1d)):
        if i == 0:
            tr = high_1d[i] - low_1d[i]
        else:
            tr = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        tr_1d.append(tr)
        if i < 14:
            atr_1d.append(np.nan)
        else:
            if i == 13:
                atr = np.mean(tr_1d[0:14])
            else:
                atr = (atr_1d[-1] * 13 + tr) / 14
            atr_1d.append(atr)
    
    atr_1d = np.array(atr_1d)
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr_14 / np.log10(14)) / np.log10((highest_high_14 - lowest_low_14))
    chop = np.where((highest_high_14 - lowest_low_14) == 0, 50, chop)  # avoid division by zero
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (20 for volume MA, 14 for chop)
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_1d_current = volume[i]  # Current volume (approximation for 4h bar)
        
        if position == 0:
            # Long setup: price at/above H3 with volume spike and trending market (Choppiness < 61.8)
            if (price >= camarilla_h3_aligned[i] and 
                vol_1d_current > 2.0 * vol_ma_1d_aligned[i] and  # Volume spike
                chop_aligned[i] < 61.8):                       # Trending market
                position = 1
                signals[i] = position_size
            # Short setup: price at/below L3 with volume spike and trending market
            elif (price <= camarilla_l3_aligned[i] and 
                  vol_1d_current > 2.0 * vol_ma_1d_aligned[i] and  # Volume spike
                  chop_aligned[i] < 61.8):                       # Trending market
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below pivot
            if price < pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above pivot
            if price > pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Camarilla_Volume_Chop"
timeframe = "4h"
leverage = 1.0