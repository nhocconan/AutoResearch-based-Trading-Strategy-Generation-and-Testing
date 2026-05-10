#!/usr/bin/env python3
# 1d_WeeklyPivot_Breakout_1wTrend_VolumeFilter
# Hypothesis: Price breaking weekly pivot R1/S1 levels on daily chart with weekly trend filter and volume confirmation.
# Weekly pivot levels provide stronger support/resistance than daily levels.
# Weekly trend filter ensures alignment with higher timeframe direction.
# Volume confirmation filters out low-participation breakouts.
# Designed for low trade frequency to minimize drag while capturing meaningful moves.

name = "1d_WeeklyPivot_Breakout_1wTrend_VolumeFilter"
timeframe = "1d"
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
    
    # Get weekly data for pivot calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly typical price and range for pivot points
    weekly_typical = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    weekly_range = df_1w['high'] - df_1w['low']
    
    # Weekly pivot R1 and S1 levels
    weekly_r1 = weekly_typical + weekly_range * 1.083 / 2
    weekly_s1 = weekly_typical - weekly_range * 1.083 / 2
    
    # Align weekly pivot levels to daily timeframe (use previous week's levels)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1.values)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1.values)
    
    # Weekly EMA for trend filter (34-period)
    weekly_ema_34 = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_ema_34_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema_34)
    
    # Volume confirmation (20-period MA on daily chart)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA (34), volume MA (20), and weekly pivot (need at least 1 week)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(weekly_ema_34_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > weekly_ema_34_aligned[i]
        downtrend = close[i] < weekly_ema_34_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Breakout conditions
        breakout_long = close[i] > weekly_r1_aligned[i]
        breakout_short = close[i] < weekly_s1_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above weekly R1 + weekly uptrend + volume spike
            if breakout_long and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly S1 + weekly downtrend + volume spike
            elif breakout_short and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks back below weekly R1 or weekly trend turns down
            if close[i] < weekly_r1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks back above weekly S1 or weekly trend turns up
            if close[i] > weekly_s1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals