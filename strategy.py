#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + 1d Volume Spike + Choppiness Regime
# - Williams %R (14) on 4h for mean-reversion extremes: long when %R < -80, short when %R > -20
# - Volume spike on 1d (current volume > 2x 20-day average) confirms institutional interest
# - Choppiness index (14) on 1d > 61.8 indicates ranging market (favorable for mean reversion)
# - Designed for 4h timeframe with selective entries to avoid overtrading
# - Target: 20-50 trades per year per symbol (80-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for volume and choppiness calculations
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Volume Spike: current volume > 2x 20-day average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * vol_ma_20)
    
    # Calculate Choppiness Index (14) on 1d
    atr_14 = pd.Series(np.sqrt((high_1d - low_1d)**2)).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_numerator = np.log10(atr_14.sum()) - np.log10(highest_high_14 - lowest_low_14)
    chop_denominator = np.log10(14)
    chop = 100 * (chop_numerator / chop_denominator) if chop_denominator != 0 else 50
    chop = pd.Series(chop).rolling(window=14, min_periods=14).mean().values
    chop_high = chop > 61.8  # Ranging market
    
    # Combine 1d filters: volume spike AND choppy market
    filter_1d = volume_spike & chop_high
    filter_1d_aligned = align_htf_to_ltf(prices, df_1d, filter_1d.astype(float))
    
    # Calculate Williams %R (14) on 4h timeframe
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    highest_high_14 = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * ((highest_high_14 - close_4h) / (highest_high_14 - lowest_low_14))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after Williams %R warmup
        # Skip if NaN in indicators
        if np.isnan(williams_r[i]) or np.isnan(filter_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        wr = williams_r[i]
        filt = filter_1d_aligned[i] > 0.5  # Boolean filter
        
        if position == 0:
            # Long entry: Williams %R oversold (< -80) + 1d filter
            if wr < -80 and filt:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought (> -20) + 1d filter
            elif wr > -20 and filt:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R rises above -50 or filter fails
            if wr > -50 or filt <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R falls below -50 or filter fails
            if wr < -50 or filt <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_1dVolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0