#!/usr/bin/env python3
"""
6h Camarilla pivot with 1d trend filter and volume confirmation
Hypothesis: Camarilla pivot levels act as strong support/resistance. 
Fade at R3/S3 (mean reversion), breakout continuation at R4/S4 (trend following). 
1d EMA50 filters for higher timeframe trend direction to avoid counter-trend trades. 
Volume confirms signal strength. Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend). 
Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_prev = np.roll(ema50_1d, 1)
    ema50_1d_prev[0] = ema50_1d[0]
    ema50_rising = ema50_1d > ema50_1d_prev
    ema50_falling = ema50_1d < ema50_1d_prev
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema50_rising_aligned = align_htf_to_ltf(prices, df_1d, ema50_rising)
    ema50_falling_aligned = align_htf_to_ltf(prices, df_1d, ema50_falling)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Camarilla pivot levels (based on previous day)
    # Calculate pivots from previous day's OHLC
    pivot = np.full(n, np.nan)
    r1 = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    r3 = np.full(n, np.nan)
    r4 = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    s2 = np.full(n, np.nan)
    s3 = np.full(n, np.nan)
    s4 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous day's data (already aligned from 1d data)
        # We'll calculate pivots using 1d data and align to 6h
        pass  # Will calculate after getting 1d OHLC
    
    # Calculate Camarilla levels from 1d OHLC
    if len(df_1d) >= 2:
        # Previous day's OHLC
        prev_close = df_1d['close'].iloc[:-1].values
        prev_high = df_1d['high'].iloc[:-1].values
        prev_low = df_1d['low'].iloc[:-1].values
        prev_open = df_1d['open'].iloc[:-1].values
        
        # Calculate pivot and levels
        pivots = (prev_high + prev_low + prev_close) / 3
        range_val = prev_high - prev_low
        
        r1_vals = pivots + range_val * 1.1 / 12
        r2_vals = pivots + range_val * 1.1 / 6
        r3_vals = pivots + range_val * 1.1 / 4
        r4_vals = pivots + range_val * 1.1 / 2
        s1_vals = pivots - range_val * 1.1 / 12
        s2_vals = pivots - range_val * 1.1 / 6
        s3_vals = pivots - range_val * 1.1 / 4
        s4_vals = pivots - range_val * 1.1 / 2
        
        # Shift to get previous day's levels for current day
        pivots = np.concatenate([ [pivots[0]], pivots[:-1] ])
        r1_vals = np.concatenate([ [r1_vals[0]], r1_vals[:-1] ])
        r2_vals = np.concatenate([ [r2_vals[0]], r2_vals[:-1] ])
        r3_vals = np.concatenate([ [r3_vals[0]], r3_vals[:-1] ])
        r4_vals = np.concatenate([ [r4_vals[0]], r4_vals[:-1] ])
        s1_vals = np.concatenate([ [s1_vals[0]], s1_vals[:-1] ])
        s2_vals = np.concatenate([ [s2_vals[0]], s2_vals[:-1] ])
        s3_vals = np.concatenate([ [s3_vals[0]], s3_vals[:-1] ])
        s4_vals = np.concatenate([ [s4_vals[0]], s4_vals[:-1] ])
        
        # Align to 6h timeframe
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivots)
        r1_aligned = align_htf_to_ltf(prices, df_1d, r1_vals)
        r2_aligned = align_htf_to_ltf(prices, df_1d, r2_vals)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3_vals)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4_vals)
        s1_aligned = align_htf_to_ltf(prices, df_1d, s1_vals)
        s2_aligned = align_htf_to_ltf(prices, df_1d, s2_vals)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3_vals)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4_vals)
    else:
        # Not enough data
        pivot_aligned = r1_aligned = r2_aligned = r3_aligned = r4_aligned = s1_aligned = s2_aligned = s3_aligned = s4_aligned = np.full(n, np.nan)
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For EMA50
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ema[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema50_rising_aligned[i]) or np.isnan(ema50_falling_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite signal or stoploss
        if position == 1:  # long position
            # Exit: price reaches S3 (mean reversion target) OR stoploss
            if (close[i] <= s3_aligned[i] or 
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches R3 (mean reversion target) OR stoploss
            if (close[i] >= r3_aligned[i] or 
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Camarilla levels + trend + volume
            # Fade at R3/S3 (mean reversion)
            fade_at_r3 = close[i] <= r3_aligned[i] and close[i] > s3_aligned[i]
            fade_at_s3 = close[i] >= s3_aligned[i] and close[i] < r3_aligned[i]
            
            # Breakout continuation at R4/S4
            breakout_r4 = close[i] >= r4_aligned[i]
            breakout_s4 = close[i] <= s4_aligned[i]
            
            # Mean reversion entries (fade at extremes)
            long_fade = fade_at_s3 and ema50_rising_aligned[i] and volume[i] > vol_ema[i] * 1.5
            short_fade = fade_at_r3 and ema50_falling_aligned[i] and volume[i] > vol_ema[i] * 1.5
            
            # Trend continuation entries (breakout)
            long_break = breakout_r4 and ema50_rising_aligned[i] and volume[i] > vol_ema[i] * 1.5
            short_break = breakout_s4 and ema50_falling_aligned[i] and volume[i] > vol_ema[i] * 1.5
            
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