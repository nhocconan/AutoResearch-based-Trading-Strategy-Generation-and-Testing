#!/usr/bin/env python3
# 12h_camarilla_pivot_1d_volume_v1
# Hypothesis: 12h Camarilla pivot levels with 1d volume confirmation. Camarilla levels act as intraday support/resistance; price reverts to mean between levels. Volume spike confirms institutional participation at key levels. Works in bull/bear markets: mean reversion in ranges, breakouts in trends. Target: 12-37 trades/year (50-150 over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v1"
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
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d candle
    # H4 = Close + 1.5*(High-Low), L4 = Close - 1.5*(High-Low)
    # H3 = Close + 1.0*(High-Low), L3 = Close - 1.0*(High-Low)
    # H2 = Close + 0.5*(High-Low), L2 = Close - 0.5*(High-Low)
    # H1 = Close + 0.25*(High-Low), L1 = Close - 0.25*(High-Low)
    pivot_range = high_1d - low_1d
    h4 = close_1d + 1.5 * pivot_range
    l4 = close_1d - 1.5 * pivot_range
    h3 = close_1d + 1.0 * pivot_range
    l3 = close_1d - 1.0 * pivot_range
    h2 = close_1d + 0.5 * pivot_range
    l2 = close_1d - 0.5 * pivot_range
    h1 = close_1d + 0.25 * pivot_range
    l1 = close_1d - 0.25 * pivot_range
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h2_aligned = align_htf_to_ltf(prices, df_1d, h2)
    l2_aligned = align_htf_to_ltf(prices, df_1d, l2)
    h1_aligned = align_htf_to_ltf(prices, df_1d, h1)
    l1_aligned = align_htf_to_ltf(prices, df_1d, l1)
    
    # 1d volume confirmation (volume > 1.5x 20-day average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirmed = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h2_aligned[i]) or np.isnan(l2_aligned[i]) or
            np.isnan(h1_aligned[i]) or np.isnan(l1_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches H3 or H4 resistance
            if close[i] >= h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches L3 or L4 support
            if close[i] <= l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price touches L1 or L2 support with volume confirmation
            if (close[i] <= l2_aligned[i]) and vol_confirmed[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price touches H1 or H2 resistance with volume confirmation
            elif (close[i] >= h2_aligned[i]) and vol_confirmed[i]:
                position = -1
                signals[i] = -0.25
    
    return signals