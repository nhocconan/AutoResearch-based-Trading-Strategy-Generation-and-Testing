#!/usr/bin/env python3
"""
6h Camarilla pivot from 1d: fade at R3/S3, breakout continuation at R4/S4
Hypothesis: Camarilla pivot levels derived from previous 1d OHLC provide strong intraday support/resistance.
In ranging markets (6h), price tends to revert from R3/S3 levels. In trending markets,
breakouts through R4/S4 with volume confirmation signal continuation. Works in both bull and bear
by fading extremes and catching breakouts. Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_fade_break_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivots (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous 1d bar
    # R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # We calculate for each 1d bar then align to 6h
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    range_1d = high_1d - low_1d
    r4 = close_1d + range_1d * 1.1 / 2
    r3 = close_1d + range_1d * 1.1 / 4
    s3 = close_1d - range_1d * 1.1 / 4
    s4 = close_1d - range_1d * 1.1 / 2
    
    # Align to 6h timeframe (shifted by 1 for previous day's levels)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h ATR for dynamic thresholds and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: above average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup
    start = 50
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(r4_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(s4_6h[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price reaches S3 (fade target) OR stoploss
            if (close[i] <= s3_6h[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches R3 (fade target) OR stoploss
            if (close[i] >= r3_6h[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            vol_filter = volume[i] > (1.5 * vol_ma[i])
            
            # Fade at R3/S3: price touches extreme level and reverses
            # Long fade: price touches or goes below S3 then closes back above it
            long_fade = (low[i] <= s3_6h[i]) and (close[i] > s3_6h[i]) and vol_filter
            # Short fade: price touches or goes above R3 then closes back below it
            short_fade = (high[i] >= r3_6h[i]) and (close[i] < r3_6h[i]) and vol_filter
            
            # Breakout continuation: price breaks R4/S4 with volume
            long_break = (close[i] > r4_6h[i]) and vol_filter
            short_break = (close[i] < s4_6h[i]) and vol_filter
            
            if long_fade or long_break:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_fade or short_break:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals