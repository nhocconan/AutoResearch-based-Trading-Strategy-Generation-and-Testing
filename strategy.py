#!/usr/bin/env python3
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
    
    # Get weekly data for pivot calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's high, low, close (1w shift for completed week)
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    # Calculate Weekly Pivot Points
    pivot_point = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    R1 = pivot_point + range_hl
    S1 = pivot_point - range_hl
    R2 = pivot_point + 2 * range_hl
    S2 = pivot_point - 2 * range_hl
    
    # Align Weekly Pivot levels to 6h timeframe
    R1_6h = align_htf_to_ltf(prices, df_1w, R1)
    S1_6h = align_htf_to_ltf(prices, df_1w, S1)
    R2_6h = align_htf_to_ltf(prices, df_1w, R2)
    S2_6h = align_htf_to_ltf(prices, df_1w, S2)
    
    # Get weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume spike: current volume > 2.5 * 8-period average (48h lookback)
    vol_ma_8 = np.full(n, np.nan)
    for i in range(8, n):
        vol_ma_8[i] = np.mean(volume[i-8:i])
    volume_spike = volume > (2.5 * vol_ma_8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for calculations
    start_idx = max(50, 8) + 1
    
    for i in range(start_idx, n):
        if (np.isnan(R1_6h[i]) or np.isnan(S1_6h[i]) or 
            np.isnan(R2_6h[i]) or np.isnan(S2_6h[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_8[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above R2 + weekly uptrend + volume spike
            if (close[i] > R2_6h[i] and close[i] > ema50_1w_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S2 + weekly downtrend + volume spike
            elif (close[i] < S2_6h[i] and close[i] < ema50_1w_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below S2 (reversal) or trend changes
            if (close[i] < S2_6h[i] or close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R2 (reversal) or trend changes
            if (close[i] > R2_6h[i] or close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R2S2_Breakout_1wTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0