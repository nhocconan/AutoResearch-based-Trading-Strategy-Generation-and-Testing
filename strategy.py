#!/usr/bin/env python3
# 6h_WeeklyPivot_Breakout_With_1dTrend
# Hypothesis: Weekly pivot levels (from previous week) act as strong support/resistance.
# Breakouts above weekly R1 or below weekly S1 with 1-day trend alignment and volume confirmation
# capture continuation moves. In bull markets, buy breakouts above R1 in uptrend.
# In bear markets, sell breakdowns below S1 in downtrend. Weekly pivots filter noise,
# 1d trend ensures alignment with higher timeframe momentum, volume avoids false breakouts.
# Works in both bull (follows strong uptrend breakouts) and bear (follows strong downtrend breakdowns).

name = "6h_WeeklyPivot_Breakout_With_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points (using previous week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly pivots to 6h timeframe (wait for weekly bar to close)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (20-period MA on 6h chart)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly pivots (need 2 weeks), EMA50_1d (50), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Breakout conditions
        breakout_above_r1 = close[i] > weekly_r1_aligned[i]
        breakdown_below_s1 = close[i] < weekly_s1_aligned[i]
        
        if position == 0:
            # Long entry: breakout above weekly R1 + uptrend + volume
            if breakout_above_r1 and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: breakdown below weekly S1 + downtrend + volume
            elif breakdown_below_s1 and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: breakdown below weekly pivot or trend reversal
            if close[i] < weekly_pivot_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: breakout above weekly pivot or trend reversal
            if close[i] > weekly_pivot_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals