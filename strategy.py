#!/usr/bin/env python3
"""
6h Weekly Pivot Direction + Daily Volume Spike
Hypothesis: Weekly pivot points act as strong support/resistance. Price breaking
above/below weekly pivot levels with daily volume confirmation indicates
institutional participation. Weekly trend filter (price vs weekly EMA20) avoids
counter-trend trades. Designed for low frequency with high conviction entries.
Works in bull (breakouts above pivot) and bear (breakdowns below pivot) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def calculate_pivot_points(high, low, close):
    """Calculate standard pivot points: P = (H+L+C)/3, R1=2P-L, S1=2P-H, etc."""
    p = (high + low + close) / 3.0
    r1 = 2 * p - low
    s1 = 2 * p - high
    r2 = p + (high - low)
    s2 = p - (high - low)
    r3 = high + 2 * (p - low)
    s3 = low - 2 * (high - p)
    return p, r1, r2, r3, s1, s2, s3

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points and trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points
    wp, wr1, wr2, wr3, ws1, ws2, ws3 = calculate_pivot_points(
        df_1w['high'].values,
        df_1w['low'].values,
        df_1w['close'].values
    )
    
    # Calculate weekly EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Get daily data for volume spike
    df_1d = get_htf_data(prices, '1d')
    
    # Align weekly levels to 6h timeframe
    wp_aligned = align_ltf_to_htf(prices, df_1w, wp)
    wr1_aligned = align_ltf_to_htf(prices, df_1w, wr1)
    wr2_aligned = align_ltf_to_htf(prices, df_1w, wr2)
    wr3_aligned = align_ltf_to_htf(prices, df_1w, wr3)
    ws1_aligned = align_ltf_to_htf(prices, df_1w, ws1)
    ws2_aligned = align_ltf_to_htf(prices, df_1w, ws2)
    ws3_aligned = align_ltf_to_htf(prices, df_1w, ws3)
    ema_20_1w_aligned = align_ltf_to_htf(prices, df_1w, ema_20_1w)
    
    # Daily volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(wp_aligned[i]) or np.isnan(wr1_aligned[i]) or np.isnan(ws1_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        weekly_pivot = wp_aligned[i]
        weekly_r1 = wr1_aligned[i]
        weekly_s1 = ws1_aligned[i]
        weekly_ema20 = ema_20_1w_aligned[i]
        
        if position == 0:
            # Long: break above weekly R1 with volume spike and above weekly EMA20
            if (price > weekly_r1 and volume_spike[i] and price > weekly_ema20):
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S1 with volume spike and below weekly EMA20
            elif (price < weekly_s1 and volume_spike[i] and price < weekly_ema20):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price breaks below weekly pivot or below weekly EMA20
            if price < weekly_pivot or price < weekly_ema20:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price breaks above weekly pivot or above weekly EMA20
            if price > weekly_pivot or price > weekly_ema20:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_Direction_Volume_Spike"
timeframe = "6h"
leverage = 1.0