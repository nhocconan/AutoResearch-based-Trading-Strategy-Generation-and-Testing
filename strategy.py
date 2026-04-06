#!/usr/bin/env python3
"""
6h Camarilla Pivot Reversal with 1d Trend Filter and Volume Confirmation v1
Hypothesis: Camarilla pivot reversals on 6h timeframe with daily trend filter capture mean reversion in ranging markets and breakout continuation in trending markets. Works in bull/bear via trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_reversal_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for pivot calculation and trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Daily pivot points for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels (based on previous day)
    r4_1d = close_1d + range_1d * 1.500
    r3_1d = close_1d + range_1d * 1.250
    r2_1d = close_1d + range_1d * 1.166
    r1_1d = close_1d + range_1d * 1.083
    s1_1d = close_1d - range_1d * 1.083
    s2_1d = close_1d - range_1d * 1.166
    s3_1d = close_1d - range_1d * 1.250
    s4_1d = close_1d - range_1d * 1.500
    
    # Daily trend filter: EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_prev = np.roll(ema50_1d, 1)
    ema50_1d_prev[0] = ema50_1d[0]
    ema50_rising = ema50_1d > ema50_1d_prev
    ema50_falling = ema50_1d < ema50_1d_prev
    
    # Align 1d data to 6h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    ema50_rising_aligned = align_htf_to_ltf(prices, df_1d, ema50_rising)
    ema50_falling_aligned = align_htf_to_ltf(prices, df_1d, ema50_falling)
    
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
    start = 50  # For daily EMA50
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(ema50_rising_aligned[i]) or np.isnan(ema50_falling_aligned[i]) or
            np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite Camarilla level or stoploss
        if position == 1:  # long position
            # Exit: price reaches S3 (take profit) OR stoploss
            if (close[i] <= s3_1d_aligned[i] or 
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches R3 (take profit) OR stoploss
            if (close[i] >= r3_1d_aligned[i] or 
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Camarilla reversal or breakout + trend + volume
            # Reversal at S3/R3 with trend alignment
            reversal_long = (low[i] <= s3_1d_aligned[i] and close[i] > s3_1d_aligned[i] and 
                           ema50_rising_aligned[i] and volume[i] > vol_ema[i] * 1.5)
            reversal_short = (high[i] >= r3_1d_aligned[i] and close[i] < r3_1d_aligned[i] and 
                            ema50_falling_aligned[i] and volume[i] > vol_ema[i] * 1.5)
            
            # Breakout continuation at S4/R4 with trend alignment
            breakout_long = (close[i] > s4_1d_aligned[i] and ema50_rising_aligned[i] and 
                           volume[i] > vol_ema[i] * 2.0)
            breakout_short = (close[i] < r4_1d_aligned[i] and ema50_falling_aligned[i] and 
                            volume[i] > vol_ema[i] * 2.0)
            
            if reversal_long or breakout_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif reversal_short or breakout_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals