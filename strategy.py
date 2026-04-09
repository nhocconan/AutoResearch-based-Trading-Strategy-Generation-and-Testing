#!/usr/bin/env python3
# 1d_camarilla_pivot_volume_chop_v1
# Hypothesis: 1d strategy using 1w Camarilla pivot levels with volume confirmation and chop regime filter.
# In ranging markets (2025+), price tends to revert from pivot support/resistance levels.
# Volume confirmation filters false touches. Chop filter ensures ranging conditions.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 30-100 total trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_pivot_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1w bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Camarilla levels (using formula: Close ± (Range * 1.1/12))
    camarilla_h5 = close_1w + (range_1w * 1.1 / 12)
    camarilla_h4 = close_1w + (range_1w * 1.1 / 6)
    camarilla_h3 = close_1w + (range_1w * 1.1 / 4)
    camarilla_l3 = close_1w - (range_1w * 1.1 / 4)
    camarilla_l4 = close_1w - (range_1w * 1.1 / 6)
    camarilla_l5 = close_1w - (range_1w * 1.1 / 12)
    
    # Align Camarilla levels to 1d timeframe
    h5_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h5)
    h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    l5_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l5)
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Choppiness index regime filter (14-period) - using 1d data
    high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    atr_14 = pd.Series(high - low).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    chop_denom = np.log10(atr_14) * np.log10(14)
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop = 100 * np.log10((high_14 - low_14) / chop_denom) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(h5_aligned[i]) or np.isnan(l5_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Chop regime: only trade when market is ranging (chop > 50)
        chop_regime = chop[i] > 50
        
        if position == 1:  # Long position
            # Exit: price moves below L3 or volume dries up
            if close[i] < l3_aligned[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves above H3 or volume dries up
            if close[i] > h3_aligned[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and chop_regime:
                # Long entry: price touches L5 with volume confirmation
                if close[i] <= l5_aligned[i] and low[i] <= l5_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price touches H5 with volume confirmation
                elif close[i] >= h5_aligned[i] and high[i] >= h5_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals