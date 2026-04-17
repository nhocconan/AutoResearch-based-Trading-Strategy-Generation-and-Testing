#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d Williams %R mean reversion + volume spike + choppiness regime filter.
Long when Williams %R < -80 (oversold), volume > 2.0x 20-period average, and chop > 61.8 (ranging market).
Short when Williams %R > -20 (overbought), volume > 2.0x 20-period average, and chop > 61.8.
Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts).
Designed to capture mean reversion in ranging markets with volume confirmation, effective in both bull and bear markets.
Williams %R identifies overextended moves, volume spike confirms participation, chop filter ensures ranging conditions.
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
    
    # Get 1d data for Williams %R and chop calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    
    # Calculate 14-period Chopiness Index: 100 * log10(sum(ATR) / (log10(n) * (HH - LL)))
    # Simplified: using true range and consolidation measure
    tr1 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values - pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    tr2 = abs(pd.Series(high_1d).rolling(window=14, min_periods=14).max().values - pd.Series(close_1d).shift(1).rolling(window=14, min_periods=14).max().values)
    tr3 = abs(pd.Series(low_1d).rolling(window=14, min_periods=14).min().values - pd.Series(close_1d).shift(1).rolling(window=14, min_periods=14).min().values)
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(true_range).rolling(window=14, min_periods=14).sum().values
    hh_ll = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values - pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (np.log10(14) * hh_ll + 1e-10))  # add small epsilon to avoid division by zero
    
    # Calculate 4h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for Williams %R, chop, and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        # Choppiness regime filter: chop > 61.8 indicates ranging market
        ranging_market = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) + volume confirmation + ranging market
            if (williams_r_aligned[i] < -80 and 
                volume_confirmed and 
                ranging_market):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) + volume confirmation + ranging market
            elif (williams_r_aligned[i] > -20 and 
                  volume_confirmed and 
                  ranging_market):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses above -50 (moving out of oversold territory)
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses below -50 (moving out of overbought territory)
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dWilliamsR_MeanReversion_Volume_Chop_Regime"
timeframe = "4h"
leverage = 1.0