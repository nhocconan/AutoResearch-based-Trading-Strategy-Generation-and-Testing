#!/usr/bin/env python3
# 6h_WeeklyTrend_Follow_With_DailyPullback
# Hypothesis: Weekly trend provides robust directional bias, while daily pullbacks offer high-probability entry points in trending markets. This strategy combines weekly trend direction with daily EMA pullbacks on the 6h chart, using volume confirmation to filter low-conviction moves. Works in both bull and bear markets by following the weekly trend.

name = "6h_WeeklyTrend_Follow_With_DailyPullback"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for EMA pullback
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily EMA20 for pullback entries
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume confirmation (24-period MA on 6h chart = 4 days)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA50 (50), daily EMA20 (20), volume MA (24)
    start_idx = max(50, 20, 24)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_20_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Price relative to daily EMA20 for pullback detection
        if i > 0:
            cross_above_ema20 = (close[i] > ema_20_1d_aligned[i]) and (close[i-1] <= ema_20_1d_aligned[i-1])
            cross_below_ema20 = (close[i] < ema_20_1d_aligned[i]) and (close[i-1] >= ema_20_1d_aligned[i-1])
        else:
            cross_above_ema20 = False
            cross_below_ema20 = False
        
        if position == 0:
            # Long entry: weekly uptrend + pullback to daily EMA20 from below + volume
            if weekly_uptrend and cross_above_ema20 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: weekly downtrend + pullback to daily EMA20 from above + volume
            elif weekly_downtrend and cross_below_ema20 and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: weekly trend breaks or reversal signal
            if not weekly_uptrend or cross_below_ema20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: weekly trend breaks or reversal signal
            if not weekly_downtrend or cross_above_ema20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals