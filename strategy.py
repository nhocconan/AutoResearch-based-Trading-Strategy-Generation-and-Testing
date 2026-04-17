#!/usr/bin/env python3
"""
1d_1w_Camarilla_R1S1_Breakout_Volume_Regime_v1
Daily strategy using weekly Camarilla pivot levels (R1, S1) with volume confirmation and chop regime filter.
Enters long when price breaks above weekly R1 with volume above average and chop > 61.8 (range).
Enters short when price breaks below weekly S1 with volume above average and chop > 61.8 (range).
Uses tight entry conditions to limit trades and avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Weekly Camarilla Pivot Levels (R1, S1) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r1 = close_1w + (range_1w * 1.1 / 12)
    s1 = close_1w - (range_1w * 1.1 / 12)
    
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # === Daily Volume for Confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Chop Index (14-period) for Regime Filter ===
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr2 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # True range for chop calculation
    true_range = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    true_range[0] = high[0] - low[0]
    tr_sum = pd.Series(true_range).rolling(window=14, min_periods=14).sum().values
    max_h = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_l = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * np.log10(tr_sum / (max_h - min_l)) / np.log10(14)
    # Handle division by zero or invalid cases
    chop = np.where((max_h - min_l) > 0, chop, 50.0)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: volume above 1.5x average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Chop regime filter: chop > 61.8 indicates ranging market (good for mean reversion at extremes)
        chop_filter = chop[i] > 61.8
        
        # Breakout conditions
        breakout_long = close[i] > r1_aligned[i]
        breakout_short = close[i] < s1_aligned[i]
        
        # Exit conditions: return to pivot level
        exit_long = close[i] < pivot[i]  # Using weekly pivot for exit
        exit_short = close[i] > pivot[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: break above R1 with volume and chop filter
            if breakout_long and vol_confirmed and chop_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: break below S1 with volume and chop filter
            elif breakout_short and vol_confirmed and chop_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price returns to weekly pivot
            if exit_long:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to weekly pivot
            if exit_short:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Camarilla_R1S1_Breakout_Volume_Regime_v1"
timeframe = "1d"
leverage = 1.0