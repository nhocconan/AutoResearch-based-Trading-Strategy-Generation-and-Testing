#!/usr/bin/env python3
# [24940] 6h_1d_camarilla_pivot_v2
# Hypothesis: 6-hour Camarilla pivot levels from 1-day data with volume confirmation.
# Long when price breaks above R4 with volume > 2x average, short when breaks below S4 with volume > 2x average.
# Exit when price returns to the pivot point or volume drops below 1.5x average.
# Uses Camarilla levels derived from previous day's range, effective in both trending and ranging markets.
# Target: 100-200 total trades over 4 years (25-50/year) with controlled risk.

import numpy as np
import pandas as pd
from mpt_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_camarilla_pivot_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from daily OHLC
    # Based on previous day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_pivot = np.full(len(close_1d), np.nan)
    camarilla_r4 = np.full(len(close_1d), np.nan)
    camarilla_s4 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        # Use previous day's data
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        
        pivot = (prev_high + prev_low + prev_close) / 3.0
        range_val = prev_high - prev_low
        
        camarilla_pivot[i] = pivot
        camarilla_r4[i] = pivot + (range_val * 1.5 / 2)  # R4 = pivot + 1.5 * (range/2)
        camarilla_s4[i] = pivot - (range_val * 1.5 / 2)  # S4 = pivot - 1.5 * (range/2)
    
    # Align Camarilla levels to 6-hour timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        
        if position == 1:  # Long
            # Exit: price returns to pivot or volume drops below 1.5x average
            if price <= pivot_aligned[i] or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price returns to pivot or volume drops below 1.5x average
            if price >= pivot_aligned[i] or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above R4 with volume expansion
            if price > r4_aligned[i] and vol_ratio > 2.0:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below S4 with volume expansion
            elif price < s4_aligned[i] and vol_ratio > 2.0:
                position = -1
                signals[i] = -0.25
    
    return signals