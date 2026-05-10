#148727
#!/usr/bin/env python3
"""
6h_WeeklyPivot_Breakout_1dTrend
Hypothesis: Price breaks above weekly R3 (long) or below weekly S3 (short) pivot levels with 1d EMA50 trend filter and volume confirmation. Weekly pivots act as strong support/resistance; breakouts with volume and trend alignment capture directional moves. Works in bull/bear by filtering trades in direction of daily trend. Target: 12-37 trades/year (50-150 total) to minimize fee drag.
"""

name = "6h_WeeklyPivot_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0

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
    
    # 1d data for EMA50 trend filter and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1w data for weekly pivot points
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R3 = H + 2*(Pivot - L) = 3*H - 2*L
    # S3 = L - 2*(H - Pivot) = 3*L - 2*H
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    weekly_r3 = 3 * high_1w - 2 * low_1w
    weekly_s3 = 3 * low_1w - 2 * high_1w
    
    # 1d EMA50 for trend filter
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    
    # 1d volume SMA20 for volume confirmation
    vol_sma20_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(df_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    
    # Align 1d indicators to 6h
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    # Align weekly pivot points to 6h (no extra delay needed for pivot points)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1w, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1w, weekly_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]) or \
           np.isnan(weekly_r3_aligned[i]) or np.isnan(weekly_s3_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 1d volume (scaled)
        # 4x 6h bars in 1d
        vol_1d_scaled = vol_sma20_1d_aligned[i] / 4.0
        volume_confirm = volume[i] > 1.5 * vol_1d_scaled
        
        # Trend and price relative to weekly pivot levels
        is_uptrend = close[i] > ema50_1d_aligned[i]
        is_downtrend = close[i] < ema50_1d_aligned[i]
        price_above_r3 = close[i] > weekly_r3_aligned[i]
        price_below_s3 = close[i] < weekly_s3_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly R3, in uptrend, with volume
            if price_above_r3 and is_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S3, in downtrend, with volume
            elif price_below_s3 and is_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price falls back below weekly R3 or trend turns down
            if not price_above_r3 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises back above weekly S3 or trend turns up
            if not price_below_s3 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals