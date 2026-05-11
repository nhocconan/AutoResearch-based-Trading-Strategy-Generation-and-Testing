#!/usr/bin/env python3
"""
1d_Camarilla_R3S3_Breakout_1wTrend_Volume
Hypothesis: Enters long when price breaks above Camarilla R3 level with weekly uptrend and volume spike; short when breaks below S3 with weekly downtrend and volume spike. Uses daily timeframe with weekly trend filter to capture multi-day moves in both bull and bear markets. Designed for low trade frequency (7-25/year) to minimize fee drag.
"""

name = "1d_Camarilla_R3S3_Breakout_1wTrend_Volume"
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
    
    # === DAILY DATA FOR CAMARILLA LEVELS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels from previous day
    # R3 = Close + 1.1*(High - Low)/2
    # S3 = Close - 1.1*(High - Low)/2
    cam_r3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    cam_s3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to daily bars (same index)
    r3_aligned = align_htf_to_ltf(prices, df_1d, cam_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, cam_s3)
    
    # === WEEKLY DATA FOR TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === VOLUME SPIKE FILTER ===
    # Volume ratio: current volume / 20-day average volume
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma20 > 0, volume / vol_ma20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers 20-day volume MA and weekly EMA50)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with weekly uptrend and volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and 
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with weekly downtrend and volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 (trend invalidation)
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price breaks above R3 (trend invalidation)
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals