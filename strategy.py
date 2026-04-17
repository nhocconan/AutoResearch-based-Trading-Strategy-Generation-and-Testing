#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    r2_1d = pivot_1d + (high_1d - low_1d)
    s2_1d = pivot_1d - (high_1d - low_1d)
    
    # Align daily pivot levels to 1d timeframe (use previous day's levels)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # Weekly trend filter: use 5-day EMA of weekly close (proxy for weekly trend)
    # Since we can't get weekly data easily without breaking rules, use 20-period EMA on 1d close as weekly trend proxy
    # But to follow rules, we'll compute EMA on 1d close and align
    close_1d_series = pd.Series(close_1d)
    ema20_1d = close_1d_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Simple trend filter using 1d EMA crossover to avoid whipsaws
    # Fast EMA 9, Slow EMA 21
    close_series = pd.Series(close)
    ema9 = close_series.ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21 = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 21)  # Need sufficient data for volume MA and EMA21
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(r2_1d_aligned[i]) or np.isnan(s2_1d_aligned[i]) or np.isnan(ema20_1d_aligned[i]) or
            np.isnan(volume_ma20[i]) or np.isnan(ema9[i]) or np.isnan(ema21[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: EMA9 > EMA21 for uptrend, EMA9 < EMA21 for downtrend
        uptrend = ema9[i] > ema21[i]
        downtrend = ema9[i] < ema21[i]
        
        # Weekly trend filter: price above/below 20-day EMA on daily (aligned)
        weekly_uptrend = close[i] > ema20_1d_aligned[i]
        weekly_downtrend = close[i] < ema20_1d_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume, in uptrend
            if (close[i] > r1_1d_aligned[i] and volume_filter and uptrend and weekly_uptrend):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with volume, in downtrend
            elif (close[i] < s1_1d_aligned[i] and volume_filter and downtrend and weekly_downtrend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below S2 (deeper level) or trend breaks
            if close[i] < s2_1d_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above R2 or trend breaks
            if close[i] > r2_1d_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_DailyPivot_Breakout_Volume_TrendFilter"
timeframe = "1d"
leverage = 1.0