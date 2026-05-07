#!/usr/bin/env python3
# 6h_WeeklyPivot_DonchianBreakout_Trend
# Hypothesis: 6-hour Donchian(10) breakout with volume confirmation and weekly pivot trend filter
# Works in bull markets via breakout momentum and in bear markets via short breakdowns
# Weekly pivot provides stronger trend filter than daily, reducing false signals
# Target: 12-37 trades per year (~50-150 over 4 years) with position size 0.25

name = "6h_WeeklyPivot_DonchianBreakout_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly pivot points (standard calculation)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    
    # Trend: above weekly pivot = uptrend, below = downtrend
    trend = pivot  # use pivot as reference level
    trend_aligned = align_htf_to_ltf(prices, df_1w, trend)
    
    # Donchian channels (10-period for more sensitivity on 6h)
    high_max = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_min = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Volume ratio: current volume / 10-period average volume
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 10  # Need 10 periods for Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(trend_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > high_max[i-1]  # Break above previous period's high
        breakout_down = close[i] < low_min[i-1]  # Break below previous period's low
        
        # Volume confirmation: volume > 1.3x average
        volume_confirm = vol_ratio[i] > 1.3
        
        # Trend filter from weekly pivot
        uptrend = close[i] > trend_aligned[i]
        downtrend = close[i] < trend_aligned[i]
        
        if position == 0:
            # Long: upward breakout + volume + uptrend
            if breakout_up and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: downward breakout + volume + downtrend
            elif breakout_down and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Donchian low or trend reversal
            if close[i] < low_min[i-1] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Donchian high or trend reversal
            if close[i] > high_max[i-1] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals