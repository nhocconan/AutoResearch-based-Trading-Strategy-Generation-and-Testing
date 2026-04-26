#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Weekly trend filter + daily Camarilla R3/S3 breakouts with volume spike confirmation. Uses 1d timeframe to minimize trade frequency (target 30-100 trades over 4 years). Weekly EMA50 trend ensures we only trade in the direction of the higher timeframe trend, reducing whipsaw. Volume spike ensures participation. Fixed size 0.25 to control risk and fees. Works in both bull and bear markets by following weekly trend.
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
    
    # Load weekly data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Load daily data for Camarilla levels (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla levels: R3, S3 from previous day
    rng = high_1d - low_1d
    camarilla_r3 = close_1d_vals + (rng * 1.1 / 4)   # R3 level
    camarilla_s3 = close_1d_vals - (rng * 1.1 / 4)   # S3 level
    
    # Align Camarilla levels to 1d timeframe (already aligned, but keep for consistency)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume spike: volume > 2.0 * 20-period mean (significant participation)
    vol_mean_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_mean_20)
    
    # Fixed position size to control trade frequency and risk
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (50 for weekly EMA, 20 for volume mean)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_mean_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        weekly_trend_up = close_val > ema_50_1w_aligned[i]
        weekly_trend_down = close_val < ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        # Entry conditions: breakout of Camarilla R3/S3 in direction of weekly trend with volume spike
        long_entry = (close_val > camarilla_r3_val) and weekly_trend_up and vol_spike
        short_entry = (close_val < camarilla_s3_val) and weekly_trend_down and vol_spike
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on mean reversion to midpoint (Camarilla center) or trend reversal
            mid_point = (camarilla_r3_val + camarilla_s3_val) / 2
            trend_reversed = close_val < ema_50_1w_aligned[i]  # weekly trend turned down
            if close_val < mid_point or trend_reversed:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on mean reversion to midpoint (Camarilla center) or trend reversal
            mid_point = (camarilla_r3_val + camarilla_s3_val) / 2
            trend_reversed = close_val > ema_50_1w_aligned[i]  # weekly trend turned up
            if close_val > mid_point or trend_reversed:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0