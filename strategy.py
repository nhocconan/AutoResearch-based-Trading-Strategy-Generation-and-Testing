#!/usr/bin/env python3
# 12h_daily_camarilla_pivot_volume_regime_v1
# Hypothesis: 12h strategy using 1d Camarilla pivot levels with volume confirmation and choppiness regime filter.
# Long: Price breaks above H4 pivot with volume > 1.5x 20-period average and CHOP > 61.8 (range regime)
# Short: Price breaks below L4 pivot with volume > 1.5x 20-period average and CHOP > 61.8 (range regime)
# Exit: Price returns to H3/L3 levels or opposite pivot break
# Uses 12h primary timeframe with 1d HTF for Camarilla pivot calculation and 1h HTF for choppiness index.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in both bull and bear markets by focusing on mean reversion in ranging conditions (CHOP > 61.8).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_daily_camarilla_pivot_volume_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Pivot = (High + Low + Close) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Range = High - Low
    range_1d = high_1d - low_1d
    # Camarilla levels
    h3_1d = pivot_1d + (range_1d * 1.1 / 4)
    l3_1d = pivot_1d - (range_1d * 1.1 / 4)
    h4_1d = pivot_1d + (range_1d * 1.1 / 2)
    l4_1d = pivot_1d - (range_1d * 1.1 / 2)
    
    # Align 1d Camarilla levels to 12h timeframe
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # Get 1h data for choppiness index (regime filter)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate True Range for 1h
    tr1 = np.abs(high_1h[1:] - low_1h[1:])
    tr2 = np.abs(high_1h[1:] - close_1h[:-1])
    tr3 = np.abs(low_1h[1:] - close_1h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with indices
    
    # Calculate ATR(14) for 1h
    atr_1h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Choppiness Index (CHOP) for 1h
    # CHOP = 100 * log10(sum(ATR14) / (max(high14) - min(low14))) / log10(14)
    atr_sum = pd.Series(atr_1h).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1h).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1h).rolling(window=14, min_periods=14).min().values
    chop_denominator = max_high - min_low
    chop_raw = 100 * np.log10(atr_sum / chop_denominator) / np.log10(14)
    chop_1h = np.where(chop_denominator > 0, chop_raw, 50.0)  # Default to 50 when denominator is 0
    
    # Align 1h Choppiness Index to 12h timeframe
    chop_1h_aligned = align_htf_to_ltf(prices, df_1h, chop_1h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period for all indicators
        # Skip if any required data is NaN
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or 
            np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(chop_1h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Regime filter: CHOP > 61.8 indicates ranging market (mean reversion favorable)
        regime_filter = chop_1h_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: Price returns to H3 level or breaks below L4 (opposite signal)
            if close[i] <= h3_1d_aligned[i] or close[i] < l4_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to L3 level or breaks above H4 (opposite signal)
            if close[i] >= l3_1d_aligned[i] or close[i] > h4_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above H4 with volume confirmation and ranging regime
            if close[i] > h4_1d_aligned[i] and volume_confirmed and regime_filter:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below L4 with volume confirmation and ranging regime
            elif close[i] < l4_1d_aligned[i] and volume_confirmed and regime_filter:
                position = -1
                signals[i] = -0.25
    
    return signals