#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: Use daily Camarilla pivot levels R3 and S3 for breakout entries with trend filter from 1d EMA34 and volume confirmation.
Designed for 12h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with strong performance in both bull and bear markets.
"""

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla pivot calculation and trend filter
    df_daily = get_htf_data(prices, '1d')
    
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (using prior day)
    high_prev = df_daily['high'].shift(1).values
    low_prev = df_daily['low'].shift(1).values
    close_prev = df_daily['close'].shift(1).values
    
    # Camarilla levels: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_r3 = close_prev + (high_prev - low_prev) * 1.1 / 4
    camarilla_s3 = close_prev - (high_prev - low_prev) * 1.1 / 4
    
    # Align daily Camarilla levels to 12h timeframe (wait for daily bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s3)
    
    # Daily EMA34 for trend filter
    ema_34 = pd.Series(df_daily['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_daily, ema_34)
    
    # Get 12h price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily Camarilla (needs 1 day), EMA34 (34 bars), volume EMA (20)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: above EMA34 (uptrend) AND price breaks above Camarilla R3 with volume
            if close[i] > ema_34_aligned[i] and high[i] > camarilla_r3_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: below EMA34 (downtrend) AND price breaks below Camarilla S3 with volume
            elif close[i] < ema_34_aligned[i] and low[i] < camarilla_s3_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Camarilla pivot point OR trend turns bearish
            # Calculate pivot point (P) = (H+L+C)/3 from prior day
            camarilla_p = (high_prev[i] + low_prev[i] + close_prev[i]) / 3 if not (np.isnan(high_prev[i]) or np.isnan(low_prev[i]) or np.isnan(close_prev[i])) else np.nan
            camarilla_p_aligned = align_htf_to_ltf(prices, df_daily, camarilla_p) if not np.isnan(camarilla_p) else np.full(n, np.nan)
            if low[i] < camarilla_p_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Camarilla pivot point OR trend turns bullish
            if high[i] > camarilla_p_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals