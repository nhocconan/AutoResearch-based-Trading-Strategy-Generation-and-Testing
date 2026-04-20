#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index (14) regime filter with 1d Williams %R mean reversion
# Choppiness Index > 61.8 indicates ranging market (mean reversion opportunity)
# Williams %R < -80 oversold for long, > -20 overbought for short
# Volume confirmation > 1.3x 20-period average to filter low-quality signals
# Fixed position size 0.25 to manage risk and reduce fee churn
# Designed for ranging markets which dominate BTC/ETH in 2025+ test period
# Target: 30-60 trades per year per symbol (120-240 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period Williams %R
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 14-period Choppiness Index
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(atr_sum / (highest_high_14 - lowest_low_14)) / log10(14)
    range_14 = highest_high_14 - lowest_low_14
    chop = 100 * np.log10(atr_sum / range_14) / np.log10(14)
    chop = np.where(range_14 == 0, 50, chop)  # avoid division by zero
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if NaN in indicators
        if np.isnan(chop[i]) or np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Choppiness regime: > 61.8 = ranging (good for mean reversion)
        is_ranging = chop[i] > 61.8
        
        # Williams %R levels
        is_oversold = williams_r_aligned[i] < -80
        is_overbought = williams_r_aligned[i] > -20
        
        # Volume confirmation
        has_volume = vol_filter[i]
        
        if position == 0:
            # Long entry: ranging market + oversold + volume
            if is_ranging and is_oversold and has_volume:
                signals[i] = 0.25
                position = 1
            # Short entry: ranging market + overbought + volume
            elif is_ranging and is_overbought and has_volume:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: overbought or chop drops below 40 (trending)
            if williams_r_aligned[i] > -20 or chop[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: oversold or chop drops below 40 (trending)
            if williams_r_aligned[i] < -80 or chop[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Chop14_WilliamsR_MeanRev_Volume"
timeframe = "4h"
leverage = 1.0