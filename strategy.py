#!/usr/bin/env python3
"""
1d_WeeklyPivot_R1S1_Breakout_With_Volume_Confirmation
Hypothesis: Daily price breaks above/below weekly pivot R1/S1 with volume confirmation and trend filter.
In bull markets, captures breakouts above weekly resistance; in bear markets, captures breakdowns below weekly support.
Uses weekly pivot levels (calculated from prior week) to avoid look-ahead, volume surge for momentum, and 1w EMA for trend filter.
Designed for 10-25 trades/year to minimize fee drag while capturing significant directional moves in BTC/ETH.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def calculate_pivot_points(high, low, close):
    """Calculate pivot points: P = (H+L+C)/3, R1 = 2P - L, S1 = 2P - H"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    return pivot, r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation (using prior week's data)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points using prior week's HLC
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate pivots for each week
    pivots, r1_levels, s1_levels = calculate_pivot_points(weekly_high, weekly_low, weekly_close)
    
    # Align weekly pivot levels to daily timeframe (use prior week's levels)
    pivots_aligned = align_ltf_to_htf(prices, df_1w, pivots)
    r1_aligned = align_ltf_to_htf(prices, df_1w, r1_levels)
    s1_aligned = align_ltf_to_htf(prices, df_1w, s1_levels)
    
    # Volume spike: >2.0x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Trend filter: 1-week EMA20
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_ltf_to_htf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 30)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        ema20 = ema_20_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above weekly R1 with volume spike and uptrend
            if price > r1 and vol_spike and price > ema20:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with volume spike and downtrend
            elif price < s1 and vol_spike and price < ema20:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price closes below weekly pivot OR trend turns down
            if price < pivots_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif price < ema20:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price closes above weekly pivot OR trend turns up
            if price > pivots_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif price > ema20:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WeeklyPivot_R1S1_Breakout_With_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0