#!/usr/bin/env python3
"""
12h_1d_Camarilla_H3L3_With_Volume_Confirmation
Hypothesis: In the 12h timeframe, buy when price touches Camarilla H3 level with volume confirmation in a ranging market (Chop > 61.8), sell when price touches L3 level with volume confirmation in a ranging market. Uses 1d timeframe for Camarilla calculation and chop filter. Designed for low trade frequency to avoid fee drag in choppy markets where mean reversion at pivot levels works well.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla and chop calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for previous day
    # H4 = C + 1.5*(H-L), H3 = C + 1.25*(H-L), H2 = C + 1.0*(H-L), H1 = C + 0.75*(H-L)
    # L1 = C - 0.75*(H-L), L2 = C - 1.0*(H-L), L3 = C - 1.25*(H-L), L4 = C - 1.5*(H-L)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range (use shift by 1 to avoid look-ahead)
    range_1d = high_1d - low_1d
    prev_close = np.roll(close_1d, 1)
    prev_close[0] = close_1d[0]  # first value same as current
    
    H3 = prev_close + 1.25 * range_1d
    L3 = prev_close - 1.25 * range_1d
    
    # Align Camarilla levels to 12h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Chopiness index for ranging market detection (using 1d data)
    # Chop = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(n)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    
    chop = 100 * (np.log10(atr_sum) - np.log10(hl_range)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike confirmation: >1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    # Start after enough data for indicators
    start_idx = max(50, 30, 14)  # volume MA, chop period
    
    for i in range(start_idx, n):
        if (np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        h3 = H3_aligned[i]
        l3 = L3_aligned[i]
        chop_val = chop_aligned[i]
        vol_spike = volume_spike[i]
        
        # Range market condition: Chop > 61.8 indicates ranging market
        is_ranging = chop_val > 61.8
        
        if position == 0:
            # Long: price touches or crosses above L3 with volume spike in ranging market
            if price >= l3 and is_ranging and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price touches or crosses below H3 with volume spike in ranging market
            elif price <= h3 and is_ranging and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price reaches H3 or chop drops below 38.2 (trending market)
            if price >= h3 or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price reaches L3 or chop drops below 38.2 (trending market)
            if price <= l3 or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_1d_Camarilla_H3L3_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0