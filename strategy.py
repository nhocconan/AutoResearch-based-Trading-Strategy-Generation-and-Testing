#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Range_Reversion_v1
Hypothesis: Price tends to reverse from Camarilla pivot levels (H3/L3) on the 4h timeframe during ranging markets.
Uses 1d for pivot calculation, 4h for entry/exit, and a 4h Choppiness Index filter to avoid trending markets.
Works in both bull and bear markets by fading extremes in range-bound conditions.
Target: 20-30 trades per year (80-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Pivot_Range_Reversion_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1D data for Camarilla pivots and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # === CAMARILLA PIVOT LEVELS (based on previous 1d bar) ===
    prev_high = np.roll(daily_high, 1)
    prev_low = np.roll(daily_low, 1)
    prev_close = np.roll(daily_close, 1)
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # H3 and L3 levels (primary reversal zones)
    h3 = pivot - (range_val * 1.1 / 4)
    l3 = pivot + (range_val * 1.1 / 4)
    
    # Align to 4h timeframe
    h3_4h = align_htf_to_ltf(prices, df_1d, h3)
    l3_4h = align_htf_to_ltf(prices, df_1d, l3)
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
    
    # === CHOPPINESS INDEX (14-period on 4h) ===
    # Chop = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(n)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_sum = np.full(n, np.nan)
    if n >= 14:
        tr_sum = np.nansum(tr[1:15])  # Skip first NaN
        atr_sum[14] = tr_sum
        for i in range(15, n):
            tr_sum = tr_sum - tr[i-1] + tr[i]
            atr_sum[i] = tr_sum
    
    max_high = np.full(n, np.nan)
    min_low = np.full(n, np.nan)
    if n >= 14:
        max_high[13] = np.max(high[0:14])
        min_low[13] = np.min(low[0:14])
        for i in range(14, n):
            max_high[i] = max(max_high[i-1], high[i])
            min_low[i] = min(min_low[i-1], low[i])
    
    chop = np.full(n, np.nan)
    for i in range(14, n):
        if not np.isnan(atr_sum[i]) and max_high[i] > min_low[i]:
            chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(14)
    
    # Market is ranging when Chop > 61.8
    ranging = chop > 61.8
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(h3_4h[i]) or np.isnan(l3_4h[i]) or 
            np.isnan(pivot_4h[i]) or np.isnan(chop[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions: price touches H3/L3 in ranging market
        touch_h3 = high[i] >= h3_4h[i] * 0.999  # Allow small slippage
        touch_l3 = low[i] <= l3_4h[i] * 1.001
        
        long_entry = touch_l3 and ranging[i]
        short_entry = touch_h3 and ranging[i]
        
        # Exit conditions: price reverts to pivot or opposite signal
        long_exit = close[i] >= pivot_4h[i] * 0.999
        short_exit = close[i] <= pivot_4h[i] * 1.001
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals