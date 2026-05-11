#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_Breakout_1dTrend_Volume_2
Hypothesis: 4h breakout above Camarilla R3 or below S3 with 1d EMA34 trend filter and volume confirmation.
Works in bull by buying breakouts in uptrend, in bear by selling breakdowns in downtrend.
Volume ensures institutional participation. Targets 20-50 trades/year (80-200 over 4 years).
"""

name = "4h_Camarilla_Pivot_Breakout_1dTrend_Volume_2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA34 ---
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # --- Camarilla Pivot Levels from previous 1d ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_r3 = np.full_like(close_1d_prev, np.nan)
    camarilla_s3 = np.full_like(close_1d_prev, np.nan)
    
    for i in range(len(close_1d_prev)):
        if i == 0:
            continue  # Need previous day
        high_prev = high_1d[i-1]
        low_prev = low_1d[i-1]
        close_prev = close_1d_prev[i-1]
        range_val = high_prev - low_prev
        camarilla_r3[i] = close_prev + range_val * 1.1 / 4
        camarilla_s3[i] = close_prev - range_val * 1.1 / 4
    
    # Align Camarilla levels to 4h
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # --- Volume Confirmation: 4h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 35  # for EMA34 and Camarilla
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        trend_up = close_4h[i] > ema34_1d_aligned[i]
        trend_down = close_4h[i] < ema34_1d_aligned[i]
        
        # Volume confirmation
        vol_ok = volume_4h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for breakouts in direction of 1d trend with volume
            if (close_4h[i] > camarilla_r3_aligned[i] and trend_up and vol_ok):
                # Breakout above R3 in uptrend
                signals[i] = 0.25
                position = 1
            elif (close_4h[i] < camarilla_s3_aligned[i] and trend_down and vol_ok):
                # Breakdown below S3 in downtrend
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: return to opposite Camarilla level or trend reversal
            if position == 1:
                # Exit long: price returns below S3 or trend turns down
                if close_4h[i] < camarilla_s3_aligned[i] or not trend_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns above R3 or trend turns up
                if close_4h[i] > camarilla_r3_aligned[i] or not trend_down:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals