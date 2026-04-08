#!/usr/bin/env python3
# 6h_1w_1d_pivots_breakout_volume_v1
# Hypothesis: Trade breakouts from weekly and daily pivot levels with volume confirmation.
# Uses weekly pivot direction as trend filter, daily pivot levels as entry/exit points,
# and volume surge for confirmation. Works in trending markets (breakouts) and ranging
# markets (fade at S3/R3). Target: 15-35 trades/year on 6h timeframe with strict entry.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_pivots_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly pivot for trend direction (calculated from previous week)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot points
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Weekly trend: price above/below weekly pivot
    weekly_trend_up = close_1w > pivot_1w
    weekly_trend_down = close_1w < pivot_1w
    
    # Align weekly trend to 6h
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up.astype(float))
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down.astype(float))
    
    # Daily pivot for entry/exit levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot points
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    r2_1d = pivot_1d + (high_1d - low_1d)
    s2_1d = pivot_1d - (high_1d - low_1d)
    r3_1d = high_1d + 2 * (pivot_1d - low_1d)
    s3_1d = low_1d - 2 * (high_1d - pivot_1d)
    r4_1d = pivot_1d + 3 * (high_1d - low_1d)
    s4_1d = pivot_1d - 3 * (high_1d - low_1d)
    
    # Align daily pivot levels to 6h
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Volume confirmation: 6h volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_trend_up_aligned[i]) or np.isnan(weekly_trend_down_aligned[i]) or
            np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 2.0 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price hits S3/S4 OR weekly trend turns down
            if (close[i] <= s3_1d_aligned[i] or close[i] <= s4_1d_aligned[i] or
                weekly_trend_down_aligned[i] > 0.5):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price hits R3/R4 OR weekly trend turns up
            if (close[i] >= r3_1d_aligned[i] or close[i] >= r4_1d_aligned[i] or
                weekly_trend_up_aligned[i] > 0.5):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: weekly trend up + price breaks R4 with volume
            if (weekly_trend_up_aligned[i] > 0.5 and close[i] > r4_1d_aligned[i] and vol_surge):
                position = 1
                signals[i] = 0.25
            # Short entry: weekly trend down + price breaks S4 with volume
            elif (weekly_trend_down_aligned[i] > 0.5 and close[i] < s4_1d_aligned[i] and vol_surge):
                position = -1
                signals[i] = -0.25
            # Fade at S3/R3 in ranging markets (weekly trend neutral)
            elif (weekly_trend_up_aligned[i] <= 0.5 and weekly_trend_down_aligned[i] <= 0.5):
                # Long near S3
                if close[i] <= s3_1d_aligned[i] * 1.005 and vol_surge:  # 0.5% buffer
                    position = 1
                    signals[i] = 0.25
                # Short near R3
                elif close[i] >= r3_1d_aligned[i] * 0.995 and vol_surge:  # 0.5% buffer
                    position = -1
                    signals[i] = -0.25
    
    return signals