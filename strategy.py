#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_1dTrend_Breakout_v1
Hypothesis: 6h Camarilla R3/S3 breakout in direction of 1d EMA34 trend with volume confirmation.
Camarilla R3/S3 levels represent stronger support/resistance than R1/S1, reducing false breakouts.
1d EMA34 trend filter ensures we only take breakouts aligned with higher timeframe momentum.
Volume confirmation adds conviction filter. Discrete sizing (0.25) limits fee drag.
Target: 50-150 total trades over 4 years (12-37/year) by requiring HTF alignment, Camarilla breakout, trend alignment, and volume.
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
    
    # Load 1d data ONCE before loop for HTF Camarilla and EMA
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
    
    # Daily EMA34 for trend filter
    close_series_1d = pd.Series(daily_close)
    ema_34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 6h timeframe (completed daily bars only)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_d_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_d_s3)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 6h volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA and 34 for EMA)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or
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
        
        # Trend filter: price above/below daily EMA34
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        if breakout_above and volume_spike and uptrend:
            # Long signal: Camarilla R3 breakout with volume, in daily uptrend
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        elif breakout_below and volume_spike and downtrend:
            # Short signal: Camarilla S3 breakout with volume, in daily downtrend
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

name = "6h_Camarilla_R3_S3_1dTrend_Breakout_v1"
timeframe = "6h"
leverage = 1.0