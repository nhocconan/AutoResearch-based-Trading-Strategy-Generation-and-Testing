#!/usr/bin/env python3
"""
6h Camarilla Pivot + Volume Confirmation
Hypothesis: Camarilla pivot levels (based on prior day's range) identify key support/resistance.
Fade at R3/S3 levels (mean reversion), breakout continuation at R4/S4 levels (trend following).
Volume confirms institutional participation at these levels.
Works in ranging markets (fade) and trending markets (breakout).
Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivots (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    range_1d = high_1d - low_1d
    # Camarilla levels: close ± (range * multiplier)
    r3 = close_1d + range_1d * 1.1 / 2
    s3 = close_1d - range_1d * 1.1 / 2
    r4 = close_1d + range_1d * 1.1
    s4 = close_1d - range_1d * 1.1
    
    # Align to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # For volume EMA
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or 
            np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse signal or stoploss
        if position == 1:  # long position
            # Exit: price reaches R4 (take profit) OR stoploss
            if (close[i] >= r4_6h[i] or 
                close[i] <= entry_price - 2.0 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches S4 (take profit) OR stoploss
            if (close[i] <= s4_6h[i] or 
                close[i] >= entry_price + 2.0 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Camarilla levels + volume
            fade_long = (close[i] <= s3_6h[i] and 
                        volume[i] > vol_ema[i] * 1.5)
            fade_short = (close[i] >= r3_6h[i] and 
                         volume[i] > vol_ema[i] * 1.5)
            breakout_long = (close[i] > r4_6h[i] and 
                            volume[i] > vol_ema[i] * 2.0)
            breakout_short = (close[i] < s4_6h[i] and 
                             volume[i] > vol_ema[i] * 2.0)
            
            if fade_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif fade_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            elif breakout_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif breakout_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals