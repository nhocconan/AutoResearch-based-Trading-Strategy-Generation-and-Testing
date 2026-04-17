#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R mean reversion with 1d volume spike and chop regime filter.
Long when Williams %R < -80 (oversold) AND 4h volume > 1.8x 20-bar average volume AND chop > 61.8 (range regime).
Short when Williams %R > -20 (overbought) AND 4h volume > 1.8x 20-bar average volume AND chop > 61.8.
Exit when Williams %R crosses above -50 (for long) or below -50 (for short).
Uses 1d for Williams %R calculation, 4h for execution and volume, 1d chop filter to avoid trends.
Designed to capture mean reversion in ranging markets with volume confirmation. Target: 30-60 trades/year per symbol.
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
    
    # Get 1d data for Williams %R and chop regime
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low + 1e-10) * -100
    
    # Calculate 1d chop regime (choppiness index)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(n)) / log10(n)
    # Simplified: use true range and rolling sum
    tr1 = np.maximum(high_1d - low_1d, 
                     np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                np.abs(low_1d - np.roll(close_1d, 1))))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    log_sum = np.log10(sum_atr14 + 1e-10)
    log_n = np.log10(14)
    chop = 100 * log_sum / log_n
    
    # Calculate 4h volume MA for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(chop_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.8x 20-bar average
        volume_confirmed = volume[i] > 1.8 * vol_ma_20[i]
        
        # Chop regime filter: only trade in ranging markets (chop > 61.8)
        chop_filter = chop_aligned[i] > 61.8
        
        # Mean reversion conditions
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        exit_long = williams_r_aligned[i] > -50
        exit_short = williams_r_aligned[i] < -50
        
        if position == 0:
            # Long: oversold with volume confirmation and chop regime
            if (oversold and volume_confirmed and chop_filter):
                signals[i] = 0.25
                position = 1
            # Short: overbought with volume confirmation and chop regime
            elif (overbought and volume_confirmed and chop_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses above -50
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses below -50
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_MeanReversion_Volume_Chop"
timeframe = "4h"
leverage = 1.0