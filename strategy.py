#!/usr/bin/env python3
"""
12h_1d_camarilla_pivot_volume_reversion
Hypothesis: Mean-reversion at daily Camarilla pivot levels on 12h timeframe.
Uses daily Camarilla pivot levels (H4, L4, H3, L3) from prior day as support/resistance.
Enters long near L3/L4 with volume confirmation, short near H3/H4 with volume confirmation.
Filters trades using 12h Choppiness Index to avoid trending markets.
Designed to work in both bull and bear markets by fading extremes at key daily levels.
Target: 25-35 trades/year (100-140 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate previous day's Camarilla pivot levels
    # H4 = Close + 1.5 * (High - Low) * 1.1/2
    # L4 = Close - 1.5 * (High - Low) * 1.1/2
    # H3 = Close + 1.0 * (High - Low) * 1.1/2
    # L3 = Close - 1.0 * (High - Low) * 1.1/2
    # Using prior day's values (shifted by 1)
    range_1d = high_1d - low_1d
    H4 = close_1d + 1.5 * range_1d * 1.1 / 2
    L4 = close_1d - 1.5 * range_1d * 1.1 / 2
    H3 = close_1d + 1.0 * range_1d * 1.1 / 2
    L3 = close_1d - 1.0 * range_1d * 1.1 / 2
    
    # Shift to use prior day's levels (avoid look-ahead)
    H4 = np.roll(H4, 1)
    L4 = np.roll(L4, 1)
    H3 = np.roll(H3, 1)
    L3 = np.roll(L3, 1)
    # First day has no prior day
    H4[0] = L4[0] = H3[0] = L3[0] = np.nan
    
    # Align pivot levels to 12h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Calculate 12h Choppiness Index for regime filter
    def calculate_choppiness(high, low, close, window=14):
        """Calculate Choppiness Index"""
        atr = []
        for i in range(len(high)):
            if i == 0:
                tr = high[i] - low[i]
            else:
                tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            atr.append(tr)
        
        atr = np.array(atr)
        atr_sum = pd.Series(atr).rolling(window=window, min_periods=window).sum().values
        
        hh = pd.Series(high).rolling(window=window, min_periods=1).max().values
        ll = pd.Series(low).rolling(window=window, min_periods=1).min().values
        range_max = hh - ll
        
        chop = 100 * np.log10(atr_sum / range_max) / np.log10(window)
        return chop
    
    chop = calculate_choppiness(high, low, close, window=14)
    
    # Volume filter: volume above 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or 
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(chop[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Choppy market filter: only trade when Choppiness > 61.8 (ranging market)
        if chop[i] <= 61.8:
            # In trending market, reduce position or stay flat
            signals[i] = 0.0
            continue
        
        # Volume confirmation: volume above average
        if volume[i] < vol_ma[i]:
            signals[i] = 0.0
            continue
        
        # Mean reversion at Camarilla levels
        # Long near L3/L4 (support)
        long_entry = False
        short_entry = False
        
        # Long conditions: price near L3 or L4 with bullish bias
        if (abs(close[i] - L3_aligned[i]) / L3_aligned[i] < 0.002 or 
            abs(close[i] - L4_aligned[i]) / L4_aligned[i] < 0.002):
            # Additional confirmation: price above L4
            if close[i] > L4_aligned[i]:
                long_entry = True
        
        # Short conditions: price near H3/H4 (resistance)
        if (abs(close[i] - H3_aligned[i]) / H3_aligned[i] < 0.002 or 
            abs(close[i] - H4_aligned[i]) / H4_aligned[i] < 0.002):
            # Additional confirmation: price below H4
            if close[i] < H4_aligned[i]:
                short_entry = True
        
        if long_entry:
            signals[i] = 0.25
        elif short_entry:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_1d_camarilla_pivot_volume_reversion"
timeframe = "12h"
leverage = 1.0