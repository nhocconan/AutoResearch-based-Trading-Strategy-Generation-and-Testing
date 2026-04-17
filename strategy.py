#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1d Camarilla pivot R1/S1 breakout + volume confirmation + 1d chop regime filter.
Long when price breaks above 1d Camarilla R1 with volume > 1.3x 20-period average and chop < 61.8 (trending).
Short when price breaks below 1d Camarilla S1 with volume > 1.3x 20-period average and chop < 61.8.
Camarilla pivots from daily timeframe provide intraday support/resistance levels that often hold.
Volume confirmation ensures breakout strength. Chop filter avoids false signals in ranging markets.
Designed to work in both bull and bear markets by trading breakouts from key daily levels with trend filter.
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
    
    # Get 1d data for Camarilla pivots and chop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day)
    def camarilla_levels(high_vals, low_vals, close_vals):
        # Typical price
        pp = (high_vals + low_vals + close_vals) / 3.0
        range_val = high_vals - low_vals
        r1 = pp + (range_val * 1.1 / 12)
        s1 = pp - (range_val * 1.1 / 12)
        return r1, s1
    
    # Shift by 1 to use previous day's levels
    r1, s1 = camarilla_levels(high_1d, low_1d, close_1d)
    r1 = np.roll(r1, 1)
    s1 = np.roll(s1, 1)
    r1[0] = np.nan
    s1[0] = np.nan
    
    # Calculate 1d Choppiness Index (CHOP)
    def choppiness_index(high_vals, low_vals, close_vals, window):
        atr = []
        for i in range(len(high_vals)):
            if i == 0:
                tr = high_vals[i] - low_vals[i]
            else:
                tr = max(high_vals[i] - low_vals[i], 
                         abs(high_vals[i] - close_vals[i-1]), 
                         abs(low_vals[i] - close_vals[i-1]))
            atr.append(tr)
        
        atr = np.array(atr)
        atr_sum = pd.Series(atr).rolling(window=window, min_periods=window).sum().values
        
        highest_high = pd.Series(high_vals).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low_vals).rolling(window=window, min_periods=window).min().values
        range_max_min = highest_high - lowest_low
        
        chop = 100 * np.log10(atr_sum / range_max_min) / np.log10(window)
        # Handle division by zero and invalid values
        chop = np.where((range_max_min == 0) | np.isnan(atr_sum) | np.isinf(atr_sum), 50, chop)
        chop = np.where(chop > 100, 100, chop)
        chop = np.where(chop < 0, 0, chop)
        return chop
    
    chop_14_1d = choppiness_index(high_1d, low_1d, close_1d, 14)
    
    # Calculate 12h volume 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    chop_14_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_14_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # need enough for volume MA and HTF indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(chop_14_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        # Chop regime filter: trending market (CHOP < 61.8)
        trending_regime = chop_14_1d_aligned[i] < 61.8
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R1 with volume and trending regime
            if (close[i] > r1_aligned[i] and 
                volume_confirmed and 
                trending_regime):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla S1 with volume and trending regime
            elif (close[i] < s1_aligned[i] and 
                  volume_confirmed and 
                  trending_regime):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 1d Camarilla S1 (opposite side)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 1d Camarilla R1 (opposite side)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dCamarilla_R1S1_Breakout_Volume_ChopFilter"
timeframe = "12h"
leverage = 1.0