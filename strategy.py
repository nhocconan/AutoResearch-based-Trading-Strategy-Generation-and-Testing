#!/usr/bin/env python3
"""
1d_1w_Camarilla_Pivot_Breakout_Volume_Filter_v2
Hypothesis: Daily Camarilla pivot levels (S3/R3) provide strong support/resistance.
Breakouts above R3 or below S3 on daily chart with volume expansion capture institutional moves.
Weekly trend filter ensures trades align with higher-timeframe momentum.
Target: 15-25 trades/year per symbol to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 300:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each daily bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's close (avoid look-ahead)
    close_prev = np.roll(close_1d, 1)
    close_prev[0] = close_1d[0]  # first bar uses its own close
    
    range_1d = high_1d - low_1d
    
    # Resistance levels (R3 used) and Support levels (S3 used)
    R3 = close_prev + (range_1d * 1.2500 / 4)
    S3 = close_prev - (range_1d * 1.2500 / 4)
    
    # Align levels to daily timeframe (1d is our primary timeframe)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    # Weekly trend filter: EMA(50) slope
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_slope = np.diff(ema_50_aligned, prepend=ema_50_aligned[0]) > 0
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(300, n):
        # Skip if any required data is not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(volume_expansion[i]) or np.isnan(ema_slope[i])):
            signals[i] = 0.0
            continue
        
        # Long breakout: price breaks above R3 with volume expansion and weekly uptrend
        long_breakout = close[i] > R3_aligned[i] and volume_expansion[i] and ema_slope[i]
        
        # Short breakdown: price breaks below S3 with volume expansion and weekly downtrend
        short_breakout = close[i] < S3_aligned[i] and volume_expansion[i] and not ema_slope[i]
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1d_1w_Camarilla_Pivot_Breakout_Volume_Filter_v2"
timeframe = "1d"
leverage = 1.0