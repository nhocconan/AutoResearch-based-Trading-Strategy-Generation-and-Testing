#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1d Williams %R mean reversion + volume confirmation + chop regime filter.
Long when Williams %R < -80 (oversold) with volume > 1.5x 20-period average and CHOP > 61.8 (ranging market).
Short when Williams %R > -20 (overbought) with volume confirmation and CHOP > 61.8.
Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts) or CHOP < 38.2 (trending regime).
Uses 12h for structure and Williams %R calculation to capture mean reversion in ranging markets.
Designed to work in both bull and bear markets by focusing on ranging conditions where mean reversion works best.
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
    
    # Calculate 1d Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 12h chop regime filter (14-period)
    # Chop = 100 * log10(sum(ATR) / (log10(n) * (highest_high - lowest_low)))
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr / (np.log10(14) * (hh_14 - ll_14)))
    # Handle division by zero and invalid values
    chop = np.where((hh_14 - ll_14) == 0, 50, chop)
    chop = np.where(np.isnan(chop), 50, chop)
    
    # Calculate 12h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for Williams %R and chop calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) with volume confirmation and chop > 61.8 (ranging)
            if (williams_r_aligned[i] < -80 and 
                volume_confirmed and 
                chop_aligned[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) with volume confirmation and chop > 61.8 (ranging)
            elif (williams_r_aligned[i] > -20 and 
                  volume_confirmed and 
                  chop_aligned[i] > 61.8):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses above -50 OR chop < 38.2 (trending regime)
            if (williams_r_aligned[i] > -50 or 
                chop_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses below -50 OR chop < 38.2 (trending regime)
            if (williams_r_aligned[i] < -50 or 
                chop_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dWilliamsR_MeanReversion_Volume_Chop_Regime"
timeframe = "12h"
leverage = 1.0