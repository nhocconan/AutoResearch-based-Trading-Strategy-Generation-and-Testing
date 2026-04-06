#!/usr/bin/env python3
"""
6h Donchian(20) breakout with 1d weekly pivot direction and volume confirmation
Hypothesis: Donchian breakouts on 6h timeframe capture momentum with directional bias from 1d weekly pivots (R4/S4 for breakouts, R3/S3 for mean reversion). Volume confirms conviction. Works in bull/bear via pivot structure. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # Get 1d data for weekly pivots and volume
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivots from prior week (using 1d data)
    # Need at least 7 days for weekly calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly high/low/close (using last 7 days)
    weekly_high = np.full(len(high_1d), np.nan)
    weekly_low = np.full(len(high_1d), np.nan)
    weekly_close = np.full(len(high_1d), np.nan)
    
    for i in range(6, len(high_1d)):  # Start from 7th day (index 6)
        weekly_high[i] = np.max(high_1d[i-6:i+1])  # Last 7 days including current
        weekly_low[i] = np.min(low_1d[i-6:i+1])
        weekly_close[i] = close_1d[i]  # Current day close
    
    # Calculate pivot points and levels
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    r2 = pp + (weekly_high - weekly_low)
    s2 = pp - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pp - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pp)
    r4 = weekly_high + 3 * (weekly_high - weekly_low)
    s4 = weekly_low - 3 * (weekly_high - weekly_low)
    
    # Align weekly pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Get 1d volume for confirmation
    volume_1d = df_1d['volume'].values
    
    # 20-period average volume on 1d
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Align volume MA to 6h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Donchian channels (20-period) from 6h data
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(30, 20)  # Need enough data for Donchian and alignments
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 6h volume > 1.5x 1d average volume (scaled)
        # Scale 1d volume to 6h: approx 1/4 of 1d volume (since 4x 6h in 1d)
        vol_threshold = vol_ma_1d_aligned[i] / 4.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below S3 OR against pivot structure (below PP)
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < s3_aligned[i] or
                close[i] < pp_aligned[i] or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price breaks above R3 OR against pivot structure (above PP)
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > r3_aligned[i] or
                close[i] > pp_aligned[i] or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 12 bars flat
            if bars_since_entry >= 12:
                # Breakout entries: R4/S4 with volume (continuation)
                bull_breakout = close[i] > r4_aligned[i]
                bear_breakout = close[i] < s4_aligned[i]
                
                # Mean reversion entries: R3/S3 with volume (fade)
                bull_reversion = close[i] < r3_aligned[i] and close[i] > pp_aligned[i]
                bear_reversion = close[i] > s3_aligned[i] and close[i] < pp_aligned[i]
                
                # Long: breakout above R4 with volume OR mean reversion from R3 with volume
                if (bull_breakout and volume_filter) or (bull_reversion and volume_filter):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown below S4 with volume OR mean reversion from S3 with volume
                elif (bear_breakout and volume_filter) or (bear_reversion and volume_filter):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals