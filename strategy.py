#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_1wTrend_Breakout_v1
Hypothesis: 12h Camarilla R3/S3 breakout in direction of 1w EMA50 trend with volume confirmation.
Uses weekly EMA for stronger trend filter, reducing whipsaws in bear markets.
Camarilla R3/S3 levels provide strong support/resistance for fewer, higher-quality breakouts.
Volume confirmation adds conviction. Discrete sizing (0.25) limits fee drag.
Target: 50-150 total trades over 4 years (12-37/year) by requiring weekly trend alignment.
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
    
    # Load 1w data ONCE before loop for HTF Camarilla and EMA
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla pivot and levels (based on previous week's OHLC)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Weekly Camarilla pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly Camarilla R3 and S3
    weekly_range = weekly_high - weekly_low
    camarilla_w_r3 = weekly_close + 1.1 * weekly_range / 4
    camarilla_w_s3 = weekly_close - 1.1 * weekly_range / 4
    
    # Weekly EMA50 for trend filter
    close_series_1w = pd.Series(weekly_close)
    ema_50_1w = close_series_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 12h timeframe (completed weekly bars only)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_w_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_w_s3)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 12h volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA and 50 for EMA)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or
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
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
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

name = "12h_Camarilla_R3_S3_1wTrend_Breakout_v1"
timeframe = "12h"
leverage = 1.0