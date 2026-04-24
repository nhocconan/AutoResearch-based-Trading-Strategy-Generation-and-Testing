#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R extreme reversal with 1d volume spike filter and chop regime.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for volume confirmation and choppiness regime detection.
- Williams %R(14): Long when < -80 (oversold), Short when > -20 (overbought).
- Regime: Choppiness Index(14) < 38.2 = trending (favor reversals at extremes), > 61.8 = choppy (avoid).
- Volume: Current 4h volume > 2.0 * 20-period average 1d volume (aligned).
- Exit: Williams %R crosses back above -50 for long exit, below -50 for short exit.
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by fading extremes only in trending regimes with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Williams %R(14)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need sufficient data for Williams %R
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R calculation: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1d Choppiness Index(14) for regime filter
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr = np.concatenate([[np.nan], tr])  # Align length
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Max(High) - Min(Low) over 14 periods
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_max_min = max_high - min_low
    
    # Choppiness Index: 100 * log10(tr_sum / range_max_min) / log10(14)
    # Avoid division by zero and log of zero
    with np.errstate(divide='ignore', invalid='ignore'):
        chop_ratio = np.where((range_max_min > 0) & (tr_sum > 0), tr_sum / range_max_min, np.nan)
        chop = 100 * np.log10(chop_ratio) / np.log10(14)
    
    # Align Choppiness Index to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20)  # Need 14 for Williams %R/Chop, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(williams_r_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Regime filter: only trade reversals in trending markets (Chop < 38.2)
        trending_regime = chop_aligned[i] < 38.2
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Exit conditions: Williams %R crosses back through -50
        if position != 0:
            # Exit long: Williams %R > -50
            if position == 1:
                if williams_r_aligned[i] > -50:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Williams %R < -50
            elif position == -1:
                if williams_r_aligned[i] < -50:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R extreme with regime and volume filters
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND trending regime AND volume confirmation
            long_condition = (williams_r_aligned[i] < -80 and 
                            trending_regime and
                            volume_confirm)
            
            # Short: Williams %R > -20 (overbought) AND trending regime AND volume confirmation
            short_condition = (williams_r_aligned[i] > -20 and 
                             trending_regime and
                             volume_confirm)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR14_Extreme_1dVolSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0