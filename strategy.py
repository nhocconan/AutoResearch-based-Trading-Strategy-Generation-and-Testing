#!/usr/bin/env python3
"""
Hypothesis: 1d timeframe with 1w Williams %R mean reversion + volume spike + chop regime filter.
Long when weekly Williams %R < -80 (oversold) + daily volume > 2x 20-day average + chop > 61.8 (ranging market).
Short when weekly Williams %R > -20 (overbought) + daily volume > 2x 20-day average + chop > 61.8 (ranging market).
Exit when Williams %R returns to -50 (mean reversion) or chop < 38.2 (trending market begins).
Designed to capture mean reversion in ranging markets during bear/range conditions (2025+) while avoiding chop whipsaws.
Williams %R identifies extreme oversold/overbought conditions; volume confirms participation; chop filter ensures ranging regime.
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
    
    # Calculate 14-period Williams %R on daily data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)) * -100
    
    # Calculate 1d chopiness index (chop) for regime filter
    # Chop = 100 * log10(sum(ATR(1) over 14) / log10((max(high14) - min(low14)) * sqrt(14)))
    tr1 = np.maximum(high_1d[1:] - low_1d[:-1], np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), np.abs(low_1d[:-1] - close_1d[1:])))
    tr1 = np.concatenate([[np.nan], tr1])  # align with close index
    atr1 = pd.Series(tr1).rolling(window=1, min_periods=1).sum().values  # ATR(1) = TR
    sum_atr1_14 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr1_14 / ((max_high_14 - min_low_14) * np.sqrt(14))) / np.log10(14)
    
    # Calculate daily volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 1d timeframe (no alignment needed as prices is already 1d)
    williams_r_aligned = williams_r  # already on 1d timeframe
    chop_aligned = chop  # already on 1d timeframe
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 30  # need enough for Williams %R and chop calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2x 20-period average
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        # Chop regime filter: only trade in ranging markets (chop > 61.8)
        chop_filter = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) + volume confirmation + chop regime
            if (williams_r_aligned[i] < -80 and 
                volume_confirmed and 
                chop_filter):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) + volume confirmation + chop regime
            elif (williams_r_aligned[i] > -20 and 
                  volume_confirmed and 
                  chop_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns to -50 OR chop < 38.2 (trending begins)
            if (williams_r_aligned[i] >= -50 or 
                chop_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to -50 OR chop < 38.2 (trending begins)
            if (williams_r_aligned[i] <= -50 or 
                chop_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1wWilliamsR_MeanReversion_Volume_Chop_Regime"
timeframe = "1d"
leverage = 1.0