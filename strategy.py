#!/usr/bin/env python3
"""
1d_Weekly_Camarilla_R3_S3_Breakout_Trend
Hypothesis: Uses weekly Camarilla R3/S3 levels as support/resistance on daily chart.
Breakouts above R3 (long) or below S3 (short) with volume confirmation and weekly trend filter.
Designed for low trade frequency (10-20/year) to minimize fee decay while capturing strong
trend continuation moves in both bull and bear markets. Weekly trend avoids counter-trend trades.
"""

name = "1d_Weekly_Camarilla_R3_S3_Breakout_Trend"
timeframe = "1d"
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
    
    # Weekly Camarilla levels (R3, S3)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot and ranges for weekly Camarilla
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    r3_1w = pivot_1w + 1.1 * range_1w / 2.0
    s3_1w = pivot_1w - 1.1 * range_1w / 2.0
    
    # Align weekly levels to daily
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Weekly trend filter: EMA50
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily volume confirmation (20-day average)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for weekly calculations
    
    for i in range(start_idx, n):
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or \
           np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.8x average
        volume_confirm = volume[i] > 1.8 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: price breaks above weekly R3 with volume and above weekly EMA50
            if close[i] > r3_aligned[i] and volume_confirm and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S3 with volume and below weekly EMA50
            elif close[i] < s3_aligned[i] and volume_confirm and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below weekly S3 or below weekly EMA50
            if close[i] < s3_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above weekly R3 or above weekly EMA50
            if close[i] > r3_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals