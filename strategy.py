#!/usr/bin/env python3
"""
12h Camarilla Pivot with Volume and Choppiness Filter
Hypothesis: At 12h timeframe, Camarilla pivot levels from 1D act as strong support/resistance. 
Price touching these levels with volume confirmation in trending regimes (Choppiness < 38.2) 
provides high-probability entries. Works in bull (bounces from L3/L4) and bear (rejections from H3/H4).
Target: 80-150 total trades over 4 years (20-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_volume_chop"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 1D data for Camarilla pivots and Choppiness (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula: Close + (High-Low) * multiplier
    rng = high_1d - low_1d
    # Resistance levels
    r3 = close_1d + rng * 1.1/2
    r4 = close_1d + rng * 1.1
    # Support levels
    s3 = close_1d - rng * 1.1/2
    s4 = close_1d - rng * 1.1
    
    # Align to 12h timeframe (previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Choppiness Index on 1D (trend detection)
    # CHOP = 100 * log10(sum(TR,14) / (max(HH,14) - min(LL,14))) / log10(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For various indicators
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Choppiness filter: only trade when trending (CHOP < 38.2)
        if chop_aligned[i] >= 38.2:
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price reaches S3 (support) or stoploss
            if (low[i] <= s3_aligned[i] or
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):  # Simple ATR proxy
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches R3 (resistance) or stoploss
            if (high[i] >= r3_aligned[i] or
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: price near Camarilla levels with volume
            # Long setup: price touches/slightly penetrates S4 but closes above S3
            long_setup = ((low[i] <= s4_aligned[i] * 1.002) and  # Allow 0.2% penetration
                         (close[i] > s3_aligned[i]) and
                         vol_filter[i])
            
            # Short setup: price touches/slightly penetrates R4 but closes below R3
            short_setup = ((high[i] >= r4_aligned[i] * 0.998) and  # Allow 0.2% penetration
                          (close[i] < r3_aligned[i]) and
                          vol_filter[i])
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals