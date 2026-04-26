#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_WeeklyTrend_Filtered_v2
Hypothesis: Daily Camarilla R3/S3 breakouts filtered by weekly EMA50 trend with volume confirmation.
Uses weekly EMA50 as HTF trend filter to ensure alignment with higher timeframe momentum (1w).
Camarilla R3/S3 levels provide stronger support/resistance than R1/S1, reducing false breakouts.
Volume confirmation adds conviction. Discrete sizing (0.25) limits fee drag.
Target: 30-100 total trades over 4 years (7-25/year) by requiring weekly trend alignment, Camarilla breakout, and volume.
Works in both bull and bear markets by only taking breakouts in direction of weekly trend.
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
    
    # Load weekly data ONCE before loop for HTF EMA trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    close_series_1w = pd.Series(df_1w['close'].values)
    ema_50_1w = close_series_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA50 to daily timeframe (completed weekly bars only)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Load daily data for Camarilla calculations
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla pivot and levels (based on previous day's OHLC)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Camarilla pivot = (daily_high + daily_low + daily_close) / 3
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    # Daily Camarilla R3 and S3
    daily_range = daily_high - daily_low
    camarilla_d_r3 = daily_close + 1.1 * daily_range / 4
    camarilla_d_s3 = daily_close - 1.1 * daily_range / 4
    
    # Align HTF indicators to daily timeframe (completed daily bars only)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_d_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_d_s3)
    
    # Daily volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA and 50 for weekly EMA)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Camarilla R3/S3 breakout conditions
        breakout_above = close[i] > camarilla_r3_aligned[i]  # Break above R3
        breakout_below = close[i] < camarilla_s3_aligned[i]   # Break below S3
        
        # Trend filter: price above/below weekly EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        if breakout_above and volume_spike and uptrend:
            # Long signal: Camarilla R3 breakout with volume, in weekly uptrend
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        elif breakout_below and volume_spike and downtrend:
            # Short signal: Camarilla S3 breakout with volume, in weekly downtrend
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R3_S3_WeeklyTrend_Filtered_v2"
timeframe = "1d"
leverage = 1.0