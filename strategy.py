#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily Williams %R (14) with 1h volume spike and 4h chop regime filter.
# Long when daily %R < -80 (oversold) + 1h volume > 2.0x 20-period average + 4h chop > 61.8 (range).
# Short when daily %R > -20 (overbought) + 1h volume > 2.0x 20-period average + 4h chop > 61.8 (range).
# Exit when daily %R crosses above -50 (long) or below -50 (short) or chop < 38.2 (trend).
# Uses discrete position size 0.25. Williams %R identifies mean reversion extremes in higher timeframe.
# Volume spike confirms institutional interest. Chop filter ensures ranging market for mean reversion.
# Target: 80-180 total trades over 4 years (20-45/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once before loop for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === Daily Indicators: Williams %R (14) ===
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Align daily Williams %R to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get 1h data once before loop for volume spike
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    volume_1h = df_1h['volume'].values
    vol_ma_20_1h = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1h, vol_ma_20_1h)
    
    # Get 4h data once before loop for chop regime
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # === 4h Indicators: Chopiness Index (14) ===
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    
    # Chopiness Index
    chop = np.where((hh_14 - ll_14) == 0, 50, 100 * np.log10(tr_sum_14 / (hh_14 - ll_14)) / np.log10(14))
    
    # Align 4h chop to 4h timeframe (same timeframe, no alignment needed)
    chop_aligned = chop  # already on 4h timeframe
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        wr = williams_r_aligned[i]
        vol_ma = vol_ma_aligned[i]
        chop_val = chop_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R crosses above -50 or chop < 38.2 (trending)
            if wr > -50 or chop_val < 38.2:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R crosses below -50 or chop < 38.2 (trending)
            if wr < -50 or chop_val < 38.2:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume filter: volume > 2.0x 20-period average (1h volume aligned to 4h)
            vol_filter = vol > 2.0 * vol_ma
            
            # Chop filter: chop > 61.8 (ranging market)
            chop_filter = chop_val > 61.8
            
            # LONG: Williams %R < -80 (oversold) with volume and chop confirmation
            if (wr < -80) and vol_filter and chop_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Williams %R > -20 (overbought) with volume and chop confirmation
            elif (wr > -20) and vol_filter and chop_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_1dWilliamsR14_1hVolumeSpike_4hChopFilter_V1"
timeframe = "4h"
leverage = 1.0