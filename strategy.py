#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian Breakout + 1d Weekly Pivot Direction + Volume Filter.
Long when price breaks above Donchian(20) high AND price > weekly pivot R1 AND volume > 1.2x 20-period average.
Short when price breaks below Donchian(20) low AND price < weekly pivot S1 AND volume > 1.2x 20-period average.
Exit on opposite Donchian break or volume drop below average.
Uses 1d for weekly pivot (calculated from prior week's OHLC), 6h for Donchian and volume.
Targets trending moves aligned with weekly structure. Designed for 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # We need to group daily data into weeks (starting Monday)
    # For simplicity, we'll use rolling weekly window (5 trading days approx)
    # but better: use actual week grouping. Since we don't have day-of-week,
    # we'll approximate with 5-day rolling and align to Friday close.
    # Instead: calculate pivot from prior 5-day window (prior week)
    if len(high_1d) >= 5:
        # Prior week's high, low, close (5 days ago to 1 day ago)
        # We'll use rolling window of 5, shifted by 1 to avoid lookahead
        high_5d = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(1).values
        low_5d = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(1).values
        close_5d = pd.Series(close_1d).rolling(window=5, min_periods=5).last().shift(1).values
        
        pp = (high_5d + low_5d + close_5d) / 3.0
        r1 = 2 * pp - low_5d
        s1 = 2 * pp - high_5d
        r2 = pp + (high_5d - low_5d)
        s2 = pp - (high_5d - low_5d)
        r3 = high_5d + 2 * (pp - low_5d)
        s3 = low_5d - 2 * (high_5d - pp)
    else:
        pp = r1 = s1 = r2 = s2 = r3 = s3 = np.full_like(close_1d, np.nan)
    
    # Align weekly pivot levels to 6h
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 6h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 6h volume filter: volume > 1.2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.2 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # warmup for 20-period indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_long = close[i] > high_20[i-1]  # break above prior period's high
        breakout_short = close[i] < low_20[i-1]  # break below prior period's low
        
        # Weekly pivot filters
        above_r1 = close[i] > r1_aligned[i]
        below_s1 = close[i] < s1_aligned[i]
        
        # Volume confirmation
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: Donchian breakout above + above weekly R1 + volume
            if breakout_long and above_r1 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown below + below weekly S1 + volume
            elif breakout_short and below_s1 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Donchian breakdown below OR volume drops below average
            if close[i] < low_20[i-1] or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Donchian breakout above OR volume drops below average
            if close[i] > high_20[i-1] or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0