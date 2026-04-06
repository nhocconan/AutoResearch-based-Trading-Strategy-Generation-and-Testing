#!/usr/bin/env python3
"""
6h Camarilla Pivot with Volume Confirmation and Trend Filter
Hypothesis: Camarilla pivot levels (R3/S3 for reversals, R4/S4 for breakouts) identify institutional support/resistance.
In ranging markets: fade at R3/S3 with volume confirmation. In trending markets: breakout continuation at R4/S4.
10-period EMA determines trend direction to avoid counter-trend trades. Volume confirms institutional participation.
Works in both bull (buy dips) and bear (sell rallies) regimes. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_volume_trend_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivot (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day
    high_prev = df_1d['high'].values
    low_prev = df_1d['low'].values
    close_prev = df_1d['close'].values
    
    pivot = (high_prev + low_prev + close_prev) / 3.0
    range_prev = high_prev - low_prev
    
    # Camarilla levels
    r4 = close_prev + range_prev * 1.1 / 2.0
    r3 = close_prev + range_prev * 1.1 / 4.0
    s3 = close_prev - range_prev * 1.1 / 4.0
    s4 = close_prev - range_prev * 1.1 / 2.0
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 10-period EMA for trend filter
    close = prices['close'].values
    ema10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema10_prev = np.roll(ema10, 1)
    ema10_prev[0] = ema10[0]
    ema10_rising = ema10 > ema10_prev
    ema10_falling = ema10 < ema10_prev
    ema10_rising_aligned = align_htf_to_ltf(prices, df_1d, ema10_rising)
    ema10_falling_aligned = align_htf_to_ltf(prices, df_1d, ema10_falling)
    
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
        if (np.isnan(ema10[i]) or np.isnan(vol_ema[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(ema10_rising_aligned[i]) or 
            np.isnan(ema10_falling_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse signal or stoploss
        if position == 1:  # long position
            # Exit: price reaches S3 (mean reversion) OR stoploss
            if (low[i] <= s3_aligned[i] or 
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches R3 (mean reversion) OR stoploss
            if (high[i] >= r3_aligned[i] or 
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries based on trend
            # In uptrend: look for long at S3 (fade) or breakout above R4
            # In downtrend: look for short at R3 (fade) or breakdown below S4
            if ema10_rising_aligned[i]:  # Uptrend
                # Fade long at S3 with volume
                fade_long = (low[i] <= s3_aligned[i] and 
                            close[i] > s3_aligned[i] and 
                            volume[i] > vol_ema[i] * 1.5)
                # Breakout long above R4 with volume
                breakout_long = (high[i] > r4_aligned[i] and 
                                close[i] > r4_aligned[i] and 
                                volume[i] > vol_ema[i] * 1.5)
                
                if fade_long or breakout_long:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
            elif ema10_falling_aligned[i]:  # Downtrend
                # Fade short at R3 with volume
                fade_short = (high[i] >= r3_aligned[i] and 
                             close[i] < r3_aligned[i] and 
                             volume[i] > vol_ema[i] * 1.5)
                # Breakdown short below S4 with volume
                breakdown_short = (low[i] < s4_aligned[i] and 
                                  close[i] < s4_aligned[i] and 
                                  volume[i] > vol_ema[i] * 1.5)
                
                if fade_short or breakdown_short:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals