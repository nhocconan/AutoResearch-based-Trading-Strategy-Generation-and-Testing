#!/usr/bin/env python3
"""
6h_WeeklyPivot_Pullback_1dTrend_VolumeConfirm
Hypothesis: On 6h timeframe, enter pullbacks to weekly pivot levels (PP, R1, S1) in the direction of 1d EMA50 trend with volume confirmation. Weekly pivots act as dynamic support/resistance that price respects. In uptrend (price > 1d EMA50), buy pullbacks to S1 or PP with volume > 1.5x median. In downtrend (price < 1d EMA50), sell rallies to R1 or PP with volume confirmation. Uses discrete sizing (0.25) to minimize fee drag. Target: 50-150 trades over 4 years.
"""

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
    
    # Calculate weekly pivot points (based on prior week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly OHLC
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_open = df_1w['open'].values
    
    # Weekly pivot: PP = (H + L + C) / 3
    weekly_pp = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly R1 = (2 * PP) - L
    weekly_r1 = (2 * weekly_pp) - weekly_low
    # Weekly S1 = (2 * PP) - H
    weekly_s1 = (2 * weekly_pp) - weekly_high
    
    # Align weekly levels to 6h (wait for weekly close)
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period median
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (vol_median * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20-period for volume median)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_pp_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Determine trend from 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Long logic: pullback to weekly S1 or PP in uptrend with volume confirmation
        long_condition = uptrend and volume_confirm[i] and (
            close[i] <= weekly_s1_aligned[i] * 1.005 or  # Allow small buffer above S1
            close[i] <= weekly_pp_aligned[i] * 1.005     # Allow small buffer above PP
        )
        # Short logic: pullback to weekly R1 or PP in downtrend with volume confirmation
        short_condition = downtrend and volume_confirm[i] and (
            close[i] >= weekly_r1_aligned[i] * 0.995 or  # Allow small buffer below R1
            close[i] >= weekly_pp_aligned[i] * 0.995     # Allow small buffer below PP
        )
        
        # Exit logic: trend reversal or opposite pivot touch
        exit_long = not uptrend or close[i] >= weekly_r1_aligned[i] * 0.995
        exit_short = not downtrend or close[i] <= weekly_s1_aligned[i] * 1.005
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_WeeklyPivot_Pullback_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0