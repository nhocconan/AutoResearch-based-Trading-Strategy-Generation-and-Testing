#!/usr/bin/env python3
"""
6h Camarilla Pivot with Volume Confirmation and Trend Filter
Hypothesis: Camarilla pivot levels (R3/S3 for reversals, R4/S4 for breakouts) work well on 6h timeframe.
Institutional traders use these levels for mean reversion and breakout strategies. Volume confirms institutional participation.
Trend filter (1d EMA50) avoids counter-trend trades. Works in both bull (buy dips at S3) and bear (sell rallies at R3).
Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for pivot points and trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_prev = np.roll(ema50_1d, 1)
    ema50_1d_prev[0] = ema50_1d[0]
    ema50_rising = ema50_1d > ema50_1d_prev
    ema50_falling = ema50_1d < ema50_1d_prev
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
    start = 50  # For EMA50
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema50_rising_aligned[i]) or np.isnan(ema50_falling_aligned[i]) or 
            np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Calculate Camarilla pivot levels from previous 1d bar
        prev_close = close_1d[i-1] if i-1 >= 0 else close_1d[0]
        prev_high = df_1d['high'].values[i-1] if i-1 >= 0 else df_1d['high'].values[0]
        prev_low = df_1d['low'].values[i-1] if i-1 >= 0 else df_1d['low'].values[0]
        
        if np.isnan(prev_close) or np.isnan(prev_high) or np.isnan(prev_low):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Camarilla pivot calculations
        pivot = (prev_high + prev_low + prev_close) / 3
        range_val = prev_high - prev_low
        
        # Resistance levels
        r3 = pivot + (range_val * 1.1 / 2)
        r4 = pivot + (range_val * 1.1)
        
        # Support levels
        s3 = pivot - (range_val * 1.1 / 2)
        s4 = pivot - (range_val * 1.1)
        
        # Check exits: reverse signal or stoploss
        if position == 1:  # long position
            # Exit: price reaches R4 (take profit) OR stoploss
            if (close[i] >= r4 or 
                close[i] <= entry_price - 2.0 * (high[i] - low[i])):  # ATR proxy
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches S4 (take profit) OR stoploss
            if (close[i] <= s4 or 
                close[i] >= entry_price + 2.0 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Camarilla + trend + volume
            # Long setup: price at S3 with bullish trend and volume
            long_setup = (close[i] <= s3 * 1.005 and  # Allow small buffer
                         ema50_rising_aligned[i] and 
                         volume[i] > vol_ema[i] * 1.5)
            
            # Short setup: price at R3 with bearish trend and volume
            short_setup = (close[i] >= r3 * 0.995 and  # Allow small buffer
                          ema50_falling_aligned[i] and 
                          volume[i] > vol_ema[i] * 1.5)
            
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