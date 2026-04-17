#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_Regime
Strategy: 12-hour Camarilla pivot breakout with volume confirmation and 1d chop regime filter.
Long: Price breaks above 1d Camarilla R1 + volume > 1.3x average + chop > 61.8 (range)
Short: Price breaks below 1d Camarilla S1 + volume > 1.3x average + chop > 61.8 (range)
Exit: Price returns to 1d Camarilla pivot (center)
Position size: 0.25
Designed to capture mean-reversion bounces in ranging markets with volume confirmation.
Timeframe: 12h
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
    
    # Calculate 1d Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate 1d Choppiness Index for regime filter
    atr_1d = []
    for i in range(len(high_1d)):
        if i == 0:
            tr = high_1d[i] - low_1d[i]
        else:
            tr = max(high_1d[i] - low_1d[i], 
                     abs(high_1d[i] - close_1d[i-1]), 
                     abs(low_1d[i] - close_1d[i-1]))
        atr_1d.append(tr)
    atr_1d = np.array(atr_1d)
    
    # Smooth ATR with 14-period EMA
    atr_ma_1d = pd.Series(atr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate sum of true ranges over 14 periods
    tr_sum_1d = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: 100 * log10(tr_sum / (atr_ma * 14)) / log10(14)
    chop_1d = 100 * np.log10(tr_sum_1d / (atr_ma_1d * 14)) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume confirmation (20-period MA on 12h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # enough for ATR and EMA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period average
        volume_filter = volume[i] > (1.3 * volume_ma20[i])
        
        # Regime filter: chop > 61.8 indicates ranging market
        regime_filter = chop_1d_aligned[i] > 61.8
        
        # Breakout conditions
        breakout_up = close[i] > r1_1d_aligned[i-1]  # break above R1
        breakout_down = close[i] < s1_1d_aligned[i-1]  # break below S1
        
        # Return to pivot (mean reversion)
        return_to_pivot = abs(close[i] - pivot_1d_aligned[i]) < 0.05 * abs(r1_1d_aligned[i] - s1_1d_aligned[i])
        
        if position == 0:
            # Long: break above R1 + volume filter + ranging market
            if breakout_up and volume_filter and regime_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 + volume filter + ranging market
            elif breakout_down and volume_filter and regime_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: return to pivot or break below S1
            if return_to_pivot or breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: return to pivot or break above R1
            if return_to_pivot or breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_Regime"
timeframe = "12h"
leverage = 1.0