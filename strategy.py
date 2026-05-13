#!/usr/bin/env python3
"""
6h_MultiLevelBreakout_1dPivot_WeeklyTrend
Hypothesis: Breakouts from 6-hour price channels filtered by daily pivot points and weekly trend direction. 
- Uses daily R1/S1 and R2/S2 pivot levels from prior day to identify key support/resistance
- Requires weekly trend filter (price above/below weekly EMA20) to align with higher timeframe bias
- Breakout confirmed with volume spike (>1.5x 20-period average) to avoid false breakouts
- Designed for 6h timeframe to capture 1-3 day swings with controlled frequency (~15-30/year)
- Works in both bull/bear markets: weekly trend filter avoids counter-trend trades, pivot levels provide structure
"""

name = "6h_MultiLevelBreakout_1dPivot_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate daily pivot points (using prior day's OHLC)
    # Standard formula: P = (H + L + C)/3, R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # First day will have NaN from roll, that's handled by min_periods later
    
    pivot_point = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot_point - prev_low
    s1 = 2 * pivot_point - prev_high
    r2 = pivot_point + (prev_high - prev_low)
    s2 = pivot_point - (prev_high - prev_low)
    
    # Calculate weekly EMA20 for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align daily pivots and weekly EMA to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 6-hour price channels (donchian-like for breakout context)
    # Use 12-period (3 days) high/low for breakout levels
    high_max_12 = pd.Series(high).rolling(window=12, min_periods=12).max().values
    low_min_12 = pd.Series(low).rolling(window=12, min_periods=12).min().values
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after enough data for indicators
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(high_max_12[i]) or
            np.isnan(low_min_12[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Break above R1 or R2 with volume + weekly uptrend
            if vol_spike and close[i] > ema_20_1w_aligned[i]:
                if close[i] > r1_aligned[i] or close[i] > r2_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # SHORT: Break below S1 or S2 with volume + weekly downtrend
            elif vol_spike and close[i] < ema_20_1w_aligned[i]:
                if close[i] < s1_aligned[i] or close[i] < s2_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Break below S1 or trend turns down
            if close[i] < s1_aligned[i] or close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Break above R1 or trend turns up
            if close[i] > r1_aligned[i] or close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals