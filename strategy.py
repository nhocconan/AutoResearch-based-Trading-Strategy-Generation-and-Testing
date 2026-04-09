#!/usr/bin/env python3
# 12h_camarilla_pivot_volume_v3
# Hypothesis: 12h strategy using 1d Camarilla pivot levels with volume confirmation and chop filter.
# In ranging markets (2025+), price tends to revert from pivot support/resistance levels.
# Volume confirmation filters false touches. Discrete sizing (0.0, ±0.25) minimizes fee churn.
# Primary timeframe: 12h, HTF: 1d for Camarilla levels and regime filter.
# Target: 50-150 total trades over 4 years by requiring pivot touch + volume spike + chop > 50.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_volume_v3"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels (using formula: Close ± (Range * 1.1/12))
    camarilla_h5 = close_1d + (range_1d * 1.1 / 12)
    camarilla_h4 = close_1d + (range_1d * 1.1 / 6)
    camarilla_h3 = close_1d + (range_1d * 1.1 / 4)
    camarilla_l3 = close_1d - (range_1d * 1.1 / 4)
    camarilla_l4 = close_1d - (range_1d * 1.1 / 6)
    camarilla_l5 = close_1d - (range_1d * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe
    h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Choppiness index regime filter (14-period) - using 1d data
    high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    atr_14 = pd.Series(high_1d - low_1d).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    chop_denom = np.log10(atr_14) * np.log10(14)
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop = 100 * np.log10((high_14 - low_14) / chop_denom) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(h5_aligned[i]) or np.isnan(l5_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Chop regime: only trade when market is ranging (chop > 50)
        chop_regime = chop_aligned[i] > 50
        
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