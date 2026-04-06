#!/usr/bin/env python3
"""
6h Donchian(20) breakout with weekly pivot direction and volume confirmation
Hypothesis: 6h Donchian breakouts capture intermediate momentum. Weekly pivot provides trend bias, volume confirms breakout strength. Works in bull (breakouts above weekly pivot) and bear (breakdowns below weekly pivot).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_weekly_pivot_vol_v1"
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
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Weekly pivot point and key levels
    pivot_weekly = np.full(len(close_weekly), np.nan)
    r3_weekly = np.full(len(close_weekly), np.nan)
    s3_weekly = np.full(len(close_weekly), np.nan)
    r4_weekly = np.full(len(close_weekly), np.nan)
    s4_weekly = np.full(len(close_weekly), np.nan)
    
    for i in range(len(close_weekly)):
        if i >= 0:  # Need at least one week
            pivot_weekly[i] = (high_weekly[i] + low_weekly[i] + close_weekly[i]) / 3.0
            r3_weekly[i] = pivot_weekly[i] + 2 * (high_weekly[i] - low_weekly[i])
            s3_weekly[i] = pivot_weekly[i] - 2 * (high_weekly[i] - low_weekly[i])
            r4_weekly[i] = pivot_weekly[i] + 3 * (high_weekly[i] - low_weekly[i])
            s4_weekly[i] = pivot_weekly[i] - 3 * (high_weekly[i] - low_weekly[i])
    
    # Align weekly levels to 6h timeframe
    pivot_weekly_aligned = align_htf_to_ltf(prices, df_weekly, pivot_weekly)
    r3_weekly_aligned = align_htf_to_ltf(prices, df_weekly, r3_weekly)
    s3_weekly_aligned = align_htf_to_ltf(prices, df_weekly, s3_weekly)
    r4_weekly_aligned = align_htf_to_ltf(prices, df_weekly, r4_weekly)
    s4_weekly_aligned = align_htf_to_ltf(prices, df_weekly, s4_weekly)
    
    # Get weekly data for volume confirmation
    volume_weekly = df_weekly['volume'].values
    
    # 4-week average volume on weekly
    vol_ma_weekly = np.full(len(volume_weekly), np.nan)
    for i in range(4, len(volume_weekly)):
        vol_ma_weekly[i] = np.mean(volume_weekly[i-4:i])
    
    # Align volume MA to 6h timeframe
    vol_ma_weekly_aligned = align_htf_to_ltf(prices, df_weekly, vol_ma_weekly)
    
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
    start = 40  # Need enough data for Donchian and alignments
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(pivot_weekly_aligned[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(vol_ma_weekly_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 6h volume > 1.5x weekly average volume (scaled)
        # Scale weekly volume to 6h: approx 1/28 of weekly volume (4*7=28 6h periods in week)
        vol_threshold = vol_ma_weekly_aligned[i] / 28.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR against weekly bias
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lower[i] or
                close[i] < pivot_weekly_aligned[i] or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR against weekly bias
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > upper[i] or
                close[i] > pivot_weekly_aligned[i] or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 6 bars flat
            if bars_since_entry >= 6:
                # Breakout entries: upper/lower with weekly bias
                bull_breakout = close[i] > upper[i]
                bear_breakout = close[i] < lower[i]
                
                # Strong bias conditions
                strong_bullish = close[i] > r3_weekly_aligned[i]
                strong_bearish = close[i] < s3_weekly_aligned[i]
                
                # Long: breakout above upper with bullish weekly bias + volume
                if bull_breakout and strong_bullish and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown below lower with bearish weekly bias + volume
                elif bear_breakout and strong_bearish and volume_filter:
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