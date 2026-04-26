#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian20_Breakout_TrendFilter_v2
Hypothesis: Weekly Camarilla pivot breakouts on 6h with 4h EMA50 trend filter and volume confirmation capture strong momentum moves while avoiding counter-trend traps. Weekly structure provides reliable support/resistance levels, and 6h timeframe balances signal quality with reasonable trade frequency (target: 50-150 trades over 4 years). Works in both bull and bear markets by trading with the dominant 4h trend.
"""

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
    
    # Load weekly data ONCE before loop for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla pivot levels (using previous week's OHLC)
    if len(df_1w) < 2:
        return np.zeros(n)
    
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    # Camarilla levels: R3/S3 for breakouts, R4/S4 for extreme breakouts
    camarilla_range = prev_week_high - prev_week_low
    r3 = prev_week_close + (camarilla_range * 1.1 / 4)
    s3 = prev_week_close - (camarilla_range * 1.1 / 4)
    r4 = prev_week_close + (camarilla_range * 1.1 / 2)
    s4 = prev_week_close - (camarilla_range * 1.1 / 2)
    
    # Align weekly Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Load 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Volume confirmation: 6h volume > 1.8x 20-period EMA
    volume_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (volume_ema * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(100, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 4h trend filter (EMA50)
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Long logic: price breaks above R3 with volume spike + in uptrend
        # Use R4 for stronger breakout confirmation when available
        long_breakout = (close[i] > r3_aligned[i] and volume_spike[i] and uptrend)
        # Short logic: price breaks below S3 with volume spike + in downtrend
        short_breakout = (close[i] < s3_aligned[i] and volume_spike[i] and downtrend)
        
        # Entry logic
        if long_breakout and position != 1:
            signals[i] = 0.25
            position = 1
        elif short_breakout and position != -1:
            signals[i] = -0.25
            position = -1
        # Add to position on strong breakouts (R4/S4) with continued volume and trend
        elif long_breakout and close[i] > r4_aligned[i] and volume_spike[i] and uptrend and position == 1:
            signals[i] = 0.25  # Maintain position
        elif short_breakout and close[i] < s4_aligned[i] and volume_spike[i] and downtrend and position == -1:
            signals[i] = -0.25  # Maintain position
        # Exit conditions: price returns to opposite level or trend weakens significantly
        elif position == 1 and (close[i] < s3_aligned[i] or not uptrend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > r3_aligned[i] or not downtrend):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Donchian20_Breakout_TrendFilter_v2"
timeframe = "6h"
leverage = 1.0