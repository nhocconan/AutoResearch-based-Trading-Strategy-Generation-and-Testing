#!/usr/bin/env python3
# 6H_WeeklyPivot_DailyTrend_VolumeFilter
# Hypothesis: Uses weekly pivot points for structure and daily trend for bias, with volume confirmation on 6h.
# Enters long when price breaks above weekly R1 in daily uptrend (close > daily EMA50) with volume > 2x 20-period average.
# Enters short when price breaks below weekly S1 in daily downtrend (close < daily EMA50) with volume confirmation.
# Exits when price returns to opposite pivot level (S1 for long, R1 for short) or trend reverses.
# Uses weekly pivot points for stronger support/resistance than daily, reducing whipsaw in choppy markets.
# Targets 12-37 trades per year on 6h timeframe with position size 0.25 to minimize fee drag.

name = "6H_WeeklyPivot_DailyTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate weekly pivot points from previous weekly bar
    prev_weekly_close = df_1w['close'].shift(1).values
    prev_weekly_high = df_1w['high'].shift(1).values
    prev_weekly_low = df_1w['low'].shift(1).values
    weekly_pivot_range = prev_weekly_high - prev_weekly_low
    r1_level = prev_weekly_close + 1.1 * weekly_pivot_range * 0.5  # R1 = C + 1.1*(H-L)/2
    s1_level = prev_weekly_close - 1.1 * weekly_pivot_range * 0.5  # S1 = C - 1.1*(H-L)/2
    
    # Align weekly pivot levels to 6h timeframe (available after weekly bar closes)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_level)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_level)
    
    # Volume filter: volume > 2x 20-period average on 6h chart
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below daily EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above weekly R1 in daily uptrend with volume spike
            if (close[i] > r1_aligned[i] and 
                price_above_ema and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly S1 in daily downtrend with volume spike
            elif (close[i] < s1_aligned[i] and 
                  price_below_ema and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to weekly S1 or trend reverses to downtrend
            if (close[i] < s1_aligned[i] or 
                price_below_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to weekly R1 or trend reverses to uptrend
            if (close[i] > r1_aligned[i] or 
                price_above_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals