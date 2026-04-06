#!/usr/bin/env python3
"""
6h Camarilla Pivot + Volume + 1d Trend Filter
Hypothesis: Camarilla pivot levels from 1d provide reliable support/resistance.
Fade at R3/S3 with volume confirmation, breakout continuation at R4/S4.
Works in both bull/bear markets as it fades extremes and captures breakouts.
Target: 50-150 trades over 4 years (12-37/year) with disciplined entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

name = "6h_camarilla_pivot_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivots and trend (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for each 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    range_1d = high_1d - low_1d
    close_prev = np.roll(close_1d, 1)
    close_prev[0] = close_1d[0]  # first bar
    
    # R levels
    r1 = close_prev + (range_1d * 1.0833)
    r2 = close_prev + (range_1d * 1.1666)
    r3 = close_prev + (range_1d * 1.2500)
    r4 = close_prev + (range_1d * 1.5000)
    
    # S levels
    s1 = close_prev - (range_1d * 1.0833)
    s2 = close_prev - (range_1d * 1.1666)
    s3 = close_prev - (range_1d * 1.2500)
    s4 = close_prev - (range_1d * 1.5000)
    
    # Pivot point
    pp = (high_1d + low_1d + close_1d) / 3.0
    
    # Trend filter: 20-period EMA on 1d close
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False).mean().values
    
    # Align all 1d levels to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    ema_20_6h = align_htf_to_ltf(prices, df_1d, ema_20)
    
    # Price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = 20  # For EMA
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(r3_6h[i]) or np.isnan(ema_20_6h[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: 20-period average
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_filter = volume[i] > vol_ma * 1.5
        else:
            volume_filter = False
        
        # Check exits
        if position == 1:  # long position
            # Exit: price reaches R3 (fade level) OR closes below EMA20
            if close[i] >= r3_6h[i] or close[i] < ema_20_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches S3 (fade level) OR closes above EMA20
            if close[i] <= s3_6h[i] or close[i] > ema_20_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Fade at R3/S3: price touches extreme level with volume
            fade_long = (close[i] <= s3_6h[i] and close[i] > s4_6h[i]) and volume_filter
            fade_short = (close[i] >= r3_6h[i] and close[i] < r4_6h[i]) and volume_filter
            
            # Breakout continuation: price breaks R4/S4 with volume and trend
            breakout_long = (close[i] > r4_6h[i]) and volume_filter and (close[i] > ema_20_6h[i])
            breakout_short = (close[i] < s4_6h[i]) and volume_filter and (close[i] < ema_20_6h[i])
            
            if fade_long or breakout_long:
                signals[i] = 0.25
                position = 1
            elif fade_short or breakout_short:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals