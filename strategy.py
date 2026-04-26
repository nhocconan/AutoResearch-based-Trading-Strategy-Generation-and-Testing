#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian_Breakout
Hypothesis: Weekly pivot levels (PP, R1, S1) act as strong support/resistance on 6h timeframe.
Breakout above weekly R1 with volume spike and price above 6h EMA50 = long.
Breakdown below weekly S1 with volume spike and price below 6h EMA50 = short.
Only trade in direction of weekly trend (price above/below weekly EMA34).
Works in bull/bear via weekly trend filter - avoids counter-trend whipsaws.
Discrete position sizing (0.25) minimizes fee drag. Target: 12-25 trades/year per symbol.
"""

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
    
    # Get weekly data for pivot points and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate weekly pivot points from previous completed weekly bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_shift = df_1w['close'].shift(1).values  # Previous week close
    
    # Weekly pivot point: PP = (High + Low + Close) / 3
    weekly_pp = (high_1w + low_1w + close_1w_shift) / 3.0
    # Weekly R1: R1 = (2 * PP) - Low
    weekly_r1 = (2 * weekly_pp) - low_1w
    # Weekly S1: S1 = (2 * PP) - High
    weekly_s1 = (2 * weekly_pp) - high_1w
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # 6h EMA50 for additional trend filter
    ema_50_6h = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Volume spike detector (20-bar volume MA on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(ema_50_6h[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_34_1w_aligned[i]
        weekly_downtrend = close[i] < ema_34_1w_aligned[i]
        
        # 6h EMA50 filter
        hma_uptrend = close[i] > ema_50_6h[i]
        hma_downtrend = close[i] < ema_50_6h[i]
        
        if position == 0:
            # Long: Break above weekly R1 with volume spike and weekly uptrend
            if (close[i] > weekly_r1_aligned[i] and volume_spike[i] and 
                weekly_uptrend and hma_uptrend):
                signals[i] = 0.25
                position = 1
            # Short: Break below weekly S1 with volume spike and weekly downtrend
            elif (close[i] < weekly_s1_aligned[i] and volume_spike[i] and 
                  weekly_downtrend and hma_downtrend):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Close below weekly R1 (failed breakout) OR weekly trend change
            if close[i] < weekly_r1_aligned[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Close above weekly S1 (failed breakdown) OR weekly trend change
            if close[i] > weekly_s1_aligned[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_Donchian_Breakout"
timeframe = "6h"
leverage = 1.0