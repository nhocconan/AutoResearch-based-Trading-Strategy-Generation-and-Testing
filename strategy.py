#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d Williams %R mean reversion + volume spike + choppiness regime filter.
Williams %R identifies overbought/oversold conditions. Long when %R < -80 (oversold) with volume spike and choppy market (CHOP > 61.8).
Short when %R > -20 (overbought) with volume spike and choppy market. Exit when %R returns to -50 (mean reversion) or volume spike reversal.
Uses 1d for Williams %R calculation (more stable) and 4h for volume confirmation and entry timing.
Designed to work in both bull and bear markets by capturing mean reversion in ranging conditions (chop filter avoids trending markets).
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
    
    # Get 1d data for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate 4h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index: CHOP = 100 * log10(sum(ATR(1)) / (n * ATR(n))) / log10(1/n)
    # Using 14-period CHOP
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]), np.abs(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.abs(high_1d[0] - low_1d[0])], tr1])  # First TR is just high-low
    atr1 = tr1  # ATR(1) is just true range
    sum_tr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    chop = 100 * np.log10(sum_tr1 / (14 * atr14)) / np.log10(1/14)
    # Handle division by zero and invalid values
    chop = np.where((atr14 == 0) | np.isnan(chop) | np.isinf(chop), 50, chop)
    
    # Align 1d indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for Williams %R and CHOP calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 2.0x 20-period average (tighter filter)
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        # Chop regime filter: only trade in choppy markets (CHOP > 61.8 = ranging)
        chop_filter = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long: Williams %R oversold (< -80) with volume confirmation and choppy market
            if (williams_r_aligned[i] < -80 and 
                volume_confirmed and 
                chop_filter):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) with volume confirmation and choppy market
            elif (williams_r_aligned[i] > -20 and 
                  volume_confirmed and 
                  chop_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns to mean (-50) or volume spike reversal
            if (williams_r_aligned[i] >= -50 or 
                (volume[i] > 2.0 * vol_ma_20[i] and williams_r_aligned[i] < williams_r_aligned[i-1])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to mean (-50) or volume spike reversal
            if (williams_r_aligned[i] <= -50 or 
                (volume[i] > 2.0 * vol_ma_20[i] and williams_r_aligned[i] > williams_r_aligned[i-1])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dWilliamsR_MeanReversion_Volume_Chop_Regime"
timeframe = "4h"
leverage = 1.0