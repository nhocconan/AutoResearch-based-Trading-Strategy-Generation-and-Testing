#!/usr/bin/env python3
"""
6h WEEKLY PIVOT BREAKOUT + VOLUME CONFIRMATION v1
Hypothesis: Weekly pivot points (calculated from prior week) act as key support/resistance.
Breakouts above weekly R1 or below weekly S1 with volume confirmation capture institutional interest.
Works in bull (breakouts with momentum) and bear (breakdowns with volume) by fading mean reversion
at extreme pivot levels (R3/S3) and continuing trends beyond R4/S4. Weekly timeframe avoids
noise while providing structural levels. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_breakout_v1"
timeframe = "6h"
leverage = 1.0

def calculate_pivots(high, low, close):
    """Calculate standard pivot points: P = (H+L+C)/3, R1=2P-L, S1=2P-H, etc."""
    p = (high + low + close) / 3.0
    r1 = 2 * p - low
    s1 = 2 * p - high
    r2 = p + (high - low)
    s2 = p - (high - low)
    r3 = high + 2 * (p - low)
    s3 = low - 2 * (high - p)
    r4 = r3 + (high - low)
    s4 = s3 - (high - low)
    return p, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivots (using prior week's data)
    weekly_p = np.full(len(weekly_high), np.nan)
    weekly_r1 = np.full(len(weekly_high), np.nan)
    weekly_r2 = np.full(len(weekly_high), np.nan)
    weekly_r3 = np.full(len(weekly_high), np.nan)
    weekly_r4 = np.full(len(weekly_high), np.nan)
    weekly_s1 = np.full(len(weekly_high), np.nan)
    weekly_s2 = np.full(len(weekly_high), np.nan)
    weekly_s3 = np.full(len(weekly_high), np.nan)
    weekly_s4 = np.full(len(weekly_high), np.nan)
    
    for i in range(1, len(weekly_high)):
        p, r1, r2, r3, r4, s1, s2, s3, s4 = calculate_pivots(
            weekly_high[i-1], weekly_low[i-1], weekly_close[i-1]
        )
        weekly_p[i] = p
        weekly_r1[i] = r1
        weekly_r2[i] = r2
        weekly_r3[i] = r3
        weekly_r4[i] = r4
        weekly_s1[i] = s1
        weekly_s2[i] = s2
        weekly_s3[i] = s3
        weekly_s4[i] = s4
    
    # Align weekly pivots to 6h timeframe
    p_aligned = align_htf_to_ltf(prices, df_weekly, weekly_p)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r2)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r3)
    r4_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r4)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s2)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s3)
    s4_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s4)
    
    # Volume filter: current volume > 1.5x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(p_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below weekly S1 or stoploss hit
            if (close[i] < s1_aligned[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above weekly R1 or stoploss hit
            if (close[i] > r1_aligned[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: price breaks above weekly R1 with volume (momentum continuation)
            # OR price rejects at weekly S3/S4 with volume (mean reversion fade)
            long_breakout = (close[i] > r1_aligned[i] and volume_filter)
            long_fade = (close[i] < s3_aligned[i] and close[i] > s4_aligned[i] and volume_filter)
            
            # Short: price breaks below weekly S1 with volume (momentum continuation)
            # OR price rejects at weekly R3/R4 with volume (mean reversion fade)
            short_breakout = (close[i] < s1_aligned[i] and volume_filter)
            short_fade = (close[i] > r3_aligned[i] and close[i] < r4_aligned[i] and volume_filter)
            
            if long_breakout or long_fade:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout or short_fade:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals