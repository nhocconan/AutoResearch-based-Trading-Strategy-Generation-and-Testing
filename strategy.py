#!/usr/bin/env python3
name = "6h_WeeklyPivotBias_VolumeSpike_1dTrend"
timeframe = "6h"
leverage = 1.0

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
    
    # Weekly pivot (using previous week's high/low/close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3, S1 = 2P - H, R1 = 2P - L
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    s1 = 2 * pivot - weekly_high
    r1 = 2 * pivot - weekly_low
    
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    
    # Weekly bias: price above/below pivot
    weekly_bull = close > pivot_aligned
    weekly_bear = close < pivot_aligned
    
    # Daily trend filter (50 EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    ema_50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    daily_uptrend = close > ema_50_aligned
    
    # Volume spike: volume > 2x 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2 * volume_ma20
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 50)  # volume MA20, EMA50
    
    for i in range(start_idx, n):
        if np.isnan(weekly_bull[i]) or np.isnan(weekly_bear[i]) or np.isnan(daily_uptrend[i]) or np.isnan(volume_spike[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: weekly bullish bias, daily uptrend, volume spike
            if weekly_bull[i] and daily_uptrend[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: weekly bearish bias, daily downtrend, volume spike
            elif weekly_bear[i] and not daily_uptrend[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: weekly bearish bias or no volume spike
            if weekly_bear[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: weekly bullish bias or no volume spike
            if weekly_bull[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals