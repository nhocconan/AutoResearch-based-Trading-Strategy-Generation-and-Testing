#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d Williams %R (14) extreme reversal + volume spike + chop regime filter.
Long when Williams %R < -80 (oversold) with volume > 2x 20-period average and chop > 61.8 (range).
Short when Williams %R > -20 (overbought) with volume > 2x 20-period average and chop > 61.8 (range).
Exit when Williams %R crosses above -50 (for long) or below -50 (for short) or chop < 38.2 (trend).
Williams %R identifies exhaustion points in ranging markets; volume spike confirms participation; chop filter ensures ranging conditions.
Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag. Uses discrete sizing 0.25.
Works in both bull (buy dips in range) and bear (sell rallies in range) by fading extremes in choppy markets.
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
    
    # Get daily data for Williams %R, volume, and chop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low + 1e-10)
    
    # Calculate daily volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily Chopiness Index (14-period)
    # ATR(14) / (sum of True Range over 14 periods) * log2(14) * 100
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    # Sum of TR over 14 periods (using rolling sum)
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * (atr_14 / tr_sum_14) * np.log2(14)
    
    # Align all to 4h
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for Williams %R and chop
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(volume_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 2.0 * vol_ma_20_1d_aligned[i]
        
        # Regime filter: chop > 61.8 indicates ranging market (good for mean reversion)
        ranging_filter = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long: Williams %R oversold (< -80) with volume spike in ranging market
            if (williams_r_aligned[i] < -80 and 
                volume_confirmed and 
                ranging_filter):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) with volume spike in ranging market
            elif (williams_r_aligned[i] > -20 and 
                  volume_confirmed and 
                  ranging_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses above -50 or market starts trending (chop < 38.2)
            if (williams_r_aligned[i] > -50 or 
                chop_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses below -50 or market starts trending (chop < 38.2)
            if (williams_r_aligned[i] < -50 or 
                chop_aligned[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dWilliamsR_Volume_Chop"
timeframe = "4h"
leverage = 1.0