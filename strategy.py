#!/usr/bin/env python3
# [24883] 4h_12h1d_camarilla_pivot_v5
# Hypothesis: 4-hour strategy using 12-hour and 1-day Camarilla pivot levels with volume confirmation.
# Long when price breaks above 12h R2 with volume > 2x average and price > 1d R2.
# Short when price breaks below 12h S2 with volume > 2x average and price < 1d S2.
# Exit when price crosses opposite 12h level OR volume falls below 1.5x average.
# Uses multiple timeframe confluence to reduce false signals and improve performance in both bull and bear markets.
# Target: 20-40 trades/year per symbol.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h1d_camarilla_pivot_v5"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12-hour and 1-day data for context
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12h Pivot (using previous 12h bar's data)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    # 12h support/resistance levels (Camarilla S2/R2)
    S2_12h = pivot_12h - (range_12h * 1.1 / 6)  # 12h S2
    R2_12h = pivot_12h + (range_12h * 1.1 / 6)  # 12h R2
    
    # Calculate 1d Pivot (using previous 1d bar's data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    # 1d support/resistance levels (Camarilla S2/R2)
    S2_1d = pivot_1d - (range_1d * 1.1 / 6)  # 1d S2
    R2_1d = pivot_1d + (range_1d * 1.1 / 6)  # 1d R2
    
    # Align indicators to 4-hour timeframe
    S2_12h_aligned = align_htf_to_ltf(prices, df_12h, S2_12h)
    R2_12h_aligned = align_htf_to_ltf(prices, df_12h, R2_12h)
    S2_1d_aligned = align_htf_to_ltf(prices, df_1d, S2_1d)
    R2_1d_aligned = align_htf_to_ltf(prices, df_1d, R2_1d)
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(S2_12h_aligned[i]) or np.isnan(R2_12h_aligned[i]) or 
            np.isnan(S2_1d_aligned[i]) or np.isnan(R2_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        S2_12h = S2_12h_aligned[i]
        R2_12h = R2_12h_aligned[i]
        S2_1d = S2_1d_aligned[i]
        R2_1d = R2_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses below 12h S2 or volume drops below 1.5x average
            if price < S2_12h or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above 12h R2 or volume drops below 1.5x average
            if price > R2_12h or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above 12h R2 with volume expansion and price > 1d R2
            if price > R2_12h and vol_ratio > 2.0 and price > R2_1d:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below 12h S2 with volume expansion and price < 1d S2
            elif price < S2_12h and vol_ratio > 2.0 and price < S2_1d:
                position = -1
                signals[i] = -0.25
    
    return signals