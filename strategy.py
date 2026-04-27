# 6h_WeeklyPivot_R3S3_Breakout_1DTrend_Volume
# Hypothesis: Combines weekly pivot points (R3/S3 levels) for structural support/resistance, 
# daily trend filter to align with higher timeframe momentum, and volume confirmation to avoid false breakouts.
# Works in bull markets by catching breakouts above R3 with uptrend and volume.
# Works in bear markets by catching breakdowns below S3 with downtrend and volume.
# Weekly pivots provide robust levels less prone to noise than daily pivots.
# Target: 50-150 trades over 4 years on 6h timeframe.

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
    
    # Get weekly data for pivot points
    df_wk = get_htf_data(prices, '1w')
    if len(df_wk) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: (H + L + C) / 3
    wk_high = df_wk['high'].values
    wk_low = df_wk['low'].values
    wk_close = df_wk['close'].values
    wk_pivot = (wk_high + wk_low + wk_close) / 3.0
    
    # Calculate R3 and S3 levels
    wk_range = wk_high - wk_low
    wk_r3 = wk_pivot + wk_range * 1.1  # R3 = Pivot + 1.1 * (High - Low)
    wk_s3 = wk_pivot - wk_range * 1.1  # S3 = Pivot - 1.1 * (High - Low)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on daily for trend filter
    ema_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_period + 1)) + 
                        ema_1d[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Get 12h data for volume filter (more stable than 6h volume)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    
    # Calculate 20-period average volume on 12h
    vol_ma_12h = np.full(len(volume_12h), np.nan)
    vol_period = 20
    for i in range(vol_period, len(volume_12h)):
        vol_ma_12h[i] = np.mean(volume_12h[i-vol_period:i])
    
    # Align all indicators to 6h timeframe
    wk_r3_aligned = align_htf_to_ltf(prices, df_wk, wk_r3)
    wk_s3_aligned = align_htf_to_ltf(prices, df_wk, wk_s3)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Volume filter: current 6h volume > 1.5x 12h average volume
    vol_ma_6h = np.full(n, np.nan)
    for i in range(vol_period, n):
        vol_ma_6h[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly pivot, daily EMA, and volume averages
    start_idx = max(50, vol_period) + 5  # buffer for calculations
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(wk_r3_aligned[i]) or np.isnan(wk_s3_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_6h[i] if vol_ma_6h[i] > 0 else 0
        
        if position == 0:
            # Long: price breaks above weekly R3 + daily uptrend + volume surge
            if (price > wk_r3_aligned[i] and 
                price > ema_1d_aligned[i] and 
                vol_ratio > 1.5):
                signals[i] = size
                position = 1
            # Short: price breaks below weekly S3 + daily downtrend + volume surge
            elif (price < wk_s3_aligned[i] and 
                  price < ema_1d_aligned[i] and 
                  vol_ratio > 1.5):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns below weekly pivot OR trend reverses
            if (price < wk_pivot[min(i // (7*24*60//6), len(wk_pivot)-1)] if i >= 7*24*60//6 else wk_pivot[0] or 
                price < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price returns above weekly pivot OR trend reverses
            if (price > wk_pivot[min(i // (7*24*60//6), len(wk_pivot)-1)] if i >= 7*24*60//6 else wk_pivot[0] or 
                price > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyPivot_R3S3_Breakout_1DTrend_Volume"
timeframe = "6h"
leverage = 1.0