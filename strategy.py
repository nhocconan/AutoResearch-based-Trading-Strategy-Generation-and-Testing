#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d Camarilla R1/S1 breakout + volume confirmation + choppiness regime filter.
Long when price breaks above 1d Camarilla R1 level with volume confirmation and choppy market (CHOP > 61.8).
Short when price breaks below 1d Camarilla S1 level with volume confirmation and choppy market (CHOP > 61.8).
Exit when price returns to the 1d Camarilla midpoint (mean reversion to pivot center).
Designed to capture mean-reversion bounces off key daily pivot levels in ranging markets, which are common in BTC/ETH during 2025 bear/range conditions.
Uses 1d timeframe for Camarilla pivot structure and 4h for entry timing and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    # Pivot = (high + low + close) / 3
    # Range = high - low
    # R1 = close + (range * 1.1 / 12)
    # S1 = close - (range * 1.1 / 12)
    # R4 = close + (range * 1.1 / 2)
    # S4 = close - (range * 1.1 / 2)
    # Midpoint for exit = (R1 + S1) / 2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    midpoint_1d = (r1_1d + s1_1d) / 2.0  # equals pivot_1d
    
    # Calculate 4h choppiness index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high, lookback=14) - min(low, lookback=14))) / log10(14)
    # CHOP > 61.8 = ranging market (good for mean reversion)
    # CHOP < 38.2 = trending market
    tr = np.maximum(high[1:] - low[1:], np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])  # align with high/low
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(pd.Series(atr14).rolling(window=14, min_periods=14).sum().values / (max_high_14 - min_low_14)) / np.log10(14)
    chop = np.where((max_high_14 - min_low_14) == 0, 50, chop_raw)  # avoid division by zero
    
    # Calculate 4h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d Camarilla levels to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    midpoint_1d_aligned = align_htf_to_ltf(prices, df_1d, midpoint_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for CHOP calculation and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(midpoint_1d_aligned[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        # Chop filter: CHOP > 61.8 indicates ranging market (good for mean reversion)
        chop_filter = chop[i] > 61.8
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R1 with volume and choppy market
            if (close[i] > r1_1d_aligned[i] and 
                volume_confirmed and 
                chop_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla S1 with volume and choppy market
            elif (close[i] < s1_1d_aligned[i] and 
                  volume_confirmed and 
                  chop_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below 1d Camarilla midpoint
            if close[i] <= midpoint_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to or above 1d Camarilla midpoint
            if close[i] >= midpoint_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dCamarilla_R1S1_Breakout_Volume_ChopFilter"
timeframe = "4h"
leverage = 1.0