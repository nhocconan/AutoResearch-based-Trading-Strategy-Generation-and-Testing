#!/usr/bin/env python3
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Get weekly data for pivot calculation
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    # Calculate weekly pivot points and R1/S1 levels
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    R1 = 2 * pivot - prev_week_low
    S1 = 2 * pivot - prev_week_high
    
    # Align weekly pivot levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1w, pivot)
    R1_6h = align_htf_to_ltf(prices, df_1w, R1)
    S1_6h = align_htf_to_ltf(prices, df_1w, S1)
    
    # Volume spike: current volume > 2.0 * 20-period average (5-day lookback)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for calculations
    start_idx = max(200, 20) + 1
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_6h[i]) or np.isnan(R1_6h[i]) or np.isnan(S1_6h[i]) or 
            np.isnan(ema200_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price above pivot + weekly uptrend + volume spike
            if (close[i] > pivot_6h[i] and close[i] > ema200_1w_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price below pivot + weekly downtrend + volume spike
            elif (close[i] < pivot_6h[i] and close[i] < ema200_1w_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below pivot or trend changes
            if (close[i] < pivot_6h[i] or close[i] < ema200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above pivot or trend changes
            if (close[i] > pivot_6h[i] or close[i] > ema200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_EMA200_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0