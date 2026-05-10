#!/usr/bin/env python3
# 6h_WeeklyPivotBreakout_1dTrend_Volume
# Hypothesis: Breakout above weekly pivot R2 or below S2 with volume surge and 1d EMA trend confirmation.
# Weekly pivots provide stronger institutional levels than daily; 1d trend filter ensures alignment with higher timeframe momentum.
# Works in bull/bear by requiring trend alignment, reducing false breakouts. Targets 15-30 trades/year.

name = "6h_WeeklyPivotBreakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate weekly pivot points using previous week's OHLC
    # Pivot = (H + L + C) / 3
    # R2 = Pivot + (H - L)
    # S2 = Pivot - (H - L)
    high_prev = np.roll(df_1w['high'].values, 1)
    low_prev = np.roll(df_1w['low'].values, 1)
    close_prev = np.roll(df_1w['close'].values, 1)
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    pivot = (high_prev + low_prev + close_prev) / 3
    weekly_r2 = pivot + (high_prev - low_prev)
    weekly_s2 = pivot - (high_prev - low_prev)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume average (24-period = ~6 days of 6h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    # Warmup: need weekly pivot (needs 1 week) + EMA34 (34) + volume MA (24)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(weekly_r2[i]) or
            np.isnan(weekly_s2[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        # Determine trend from 1d EMA34
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
        uptrend = close_1d_aligned[i] > ema_34_1d_aligned[i]
        downtrend = close_1d_aligned[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation (1.5x average)
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        # Breakout above weekly R2 or breakdown below S2
        breakout_r2 = close[i] > weekly_r2[i]
        breakdown_s2 = close[i] < weekly_s2[i]
        
        if position == 0:
            bars_since_entry = 0
            # Long: Breakout above weekly R2 with volume surge and 1d uptrend
            if breakout_r2 and volume_surge and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below weekly S2 with volume surge and 1d downtrend
            elif breakdown_s2 and volume_surge and downtrend:
                signals[i] = -0.25
                position = -1
        else:
            bars_since_entry += 1
            # Enforce minimum holding period of 2 bars (12 hours)
            if bars_since_entry < 2:
                signals[i] = signals[i-1]  # maintain position
                continue
            
            if position == 1:
                # Long exit: price breaks below weekly S2 or trend changes
                if close[i] < weekly_s2[i] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: price breaks above weekly R2 or trend changes
                if close[i] > weekly_r2[i] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals