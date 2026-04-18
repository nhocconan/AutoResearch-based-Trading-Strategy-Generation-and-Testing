#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1_S1_Volume_Regime
Based on top performer: Camarilla pivot levels (R1, S1) from 1d + volume spike + Choppiness regime filter.
- Long when price breaks above R1 with volume spike and CHOP > 61.8 (range) -> mean reversion to Pivot
- Short when price breaks below S1 with volume spike and CHOP > 61.8 (range) -> mean reversion to Pivot
- Exit when price touches Pivot point (mean reversion target) or opposite breakout
- Uses 1d Camarilla levels for structure, 4h for entry timing
- Designed for 20-50 trades/year per symbol
Works in ranging markets (2025-2026) by fading false breakouts with volume confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period."""
    typical = (high + low + close) / 3
    range_val = high - low
    
    pivot = typical
    r1 = close + (range_val * 1.1 / 12)
    s1 = close - (range_val * 1.1 / 12)
    r2 = close + (range_val * 1.1 / 6)
    s2 = close - (range_val * 1.1 / 6)
    r3 = close + (range_val * 1.1 / 4)
    s3 = close - (range_val * 1.1 / 4)
    r4 = close + (range_val * 1.1 / 2)
    s4 = close - (range_val * 1.1 / 2)
    
    return pivot, r1, s1, r2, s2, r3, s3, r4, s4

def calculate_choppiness(high, low, close, window=14):
    """Calculate Choppiness Index (high values = ranging, low = trending)."""
    n = len(high)
    chop = np.full(n, np.nan)
    
    for i in range(window-1, n):
        # True Range
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1]) if i > 0 else 0
        tr3 = abs(low[i] - close[i-1]) if i > 0 else 0
        tr = max(tr1, tr2, tr3)
        
        # Sum of true ranges
        atr_sum = 0
        for j in range(i-window+1, i+1):
            tr1_j = high[j] - low[j]
            tr2_j = abs(high[j] - close[j-1]) if j > 0 else 0
            tr3_j = abs(low[j] - close[j-1]) if j > 0 else 0
            tr_j = max(tr1_j, tr2_j, tr3_j)
            atr_sum += tr_j
        
        # Avoid division by zero
        if atr_sum == 0:
            chop[i] = 50
        else:
            # Choppiness formula: 100 * log10(sum(tr) / (ATR * n)) / log10(n)
            chop[i] = 100 * (np.log10(atr_sum) - np.log10(tr * window)) / np.log10(window)
    
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels
    pivot_1d, r1_1d, s1_1d, r2_1d, s2_1d, r3_1d, s3_1d, r4_1d, s4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align 1d Camarilla levels to 4h timeframe
    pivot_1d_4h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_4h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_4h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate 4h volume spike (volume > 1.5 * 20-period average)
    volume_ma = np.zeros(n)
    volume_sum = 0
    for i in range(n):
        volume_sum += volume[i]
        if i >= 20:
            volume_sum -= volume[i-20]
        if i >= 19:
            volume_ma[i] = volume_sum / 20
        else:
            volume_ma[i] = volume_sum / (i+1) if i+1 > 0 else 0
    
    volume_spike = volume > (volume_ma * 1.5)
    
    # Calculate 4h Choppiness Index
    chop = calculate_choppiness(high, low, close, window=14)
    chop_range = chop > 61.8  # ranging market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # need volume MA and chop data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_1d_4h[i]) or np.isnan(r1_1d_4h[i]) or np.isnan(s1_1d_4h[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Price action
        price = close[i]
        prev_price = close[i-1] if i > 0 else price
        
        # Breakout conditions
        breakout_above_r1 = price > r1_1d_4h[i] and prev_price <= r1_1d_4h[i]
        breakout_below_s1 = price < s1_1d_4h[i] and prev_price >= s1_1d_4h[i]
        
        # Mean reversion conditions
        near_pivot = abs(price - pivot_1d_4h[i]) < (r1_1d_4h[i] - s1_1d_4h[i]) * 0.05  # within 5% of pivot range
        
        if position == 0:
            # Long: fade breakdown below S1 in ranging market with volume spike
            if breakout_below_s1 and chop_range[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: fade breakout above R1 in ranging market with volume spike
            elif breakout_above_r1 and chop_range[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot (mean reversion target) or breaks above R1
            if near_pivot or breakout_above_r1:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot (mean reversion target) or breaks below S1
            if near_pivot or breakout_below_s1:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_R1_S1_Volume_Regime"
timeframe = "4h"
leverage = 1.0