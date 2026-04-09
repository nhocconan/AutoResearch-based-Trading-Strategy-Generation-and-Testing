#!/usr/bin/env python3
# 1d_1w_camarilla_breakout_v1
# Hypothesis: Daily breakout of weekly Camarilla levels with volume confirmation.
# Long when price breaks above weekly H4 with volume > 1.5x 20-day average.
# Short when price breaks below weekly L4 with volume > 1.5x 20-day average.
# Exit when price returns to opposite weekly Camarilla level (L4 for longs, H4 for shorts).
# Position size fixed at 0.25 to limit drawdown. Target: 30-100 total trades over 4 years (7-25/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels
    # Camarilla: H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    camarilla_h4_1w = np.full(len(df_1w), np.nan)
    camarilla_l4_1w = np.full(len(df_1w), np.nan)
    for i in range(len(df_1w)):
        c = df_1w['close'].iloc[i]
        h = df_1w['high'].iloc[i]
        l = df_1w['low'].iloc[i]
        camarilla_h4_1w[i] = c + 1.1 * (h - l) / 2
        camarilla_l4_1w[i] = c - 1.1 * (h - l) / 2
    
    # Align weekly Camarilla levels to daily timeframe
    camarilla_h4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4_1w)
    camarilla_l4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4_1w)
    
    # Volume confirmation: 20-day average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_1w_aligned[i]) or 
            np.isnan(camarilla_l4_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below weekly L4 level
            if close[i] <= camarilla_l4_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above weekly H4 level
            if close[i] >= camarilla_h4_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above weekly H4 with volume filter
            if (close[i] > camarilla_h4_1w_aligned[i] and 
                volume[i] > vol_ma_20[i] * 1.5):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below weekly L4 with volume filter
            elif (close[i] < camarilla_l4_1w_aligned[i] and 
                  volume[i] > vol_ma_20[i] * 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals