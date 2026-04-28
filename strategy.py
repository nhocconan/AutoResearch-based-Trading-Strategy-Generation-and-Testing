#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Fade_1wTrend_VolumeFilter
Hypothesis: On 6-hour chart, price often reverts to the mean after reaching extreme Camarilla levels (R3/S3) during weekly trends. 
Fading R3/S3 with weekly trend filter and volume exhaustion filter captures mean reversion in both bull and bear markets.
Weekly trend ensures we fade in direction of higher timeframe momentum, reducing counter-trend risk.
Volume exhaustion (low volume) confirms lack of follow-through, increasing fade probability.
Target: 15-25 trades/year (60-100 total over 4 years).
"""

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's range
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    # S4 = close - 1.5 * (high - low)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    r3 = prev_close + 1.1 * (prev_high - prev_low)
    s3 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume exhaustion: volume < 0.7 * 20-period MA (low volume = lack of follow-through)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Fade conditions: price at extreme levels with volume exhaustion
        at_r3 = high[i] >= r3_aligned[i]  # touched or exceeded R3
        at_s3 = low[i] <= s3_aligned[i]   # touched or exceeded S3
        vol_exhausted = volume[i] < (0.7 * vol_ma_20[i])
        
        # Entry logic: fade extreme levels in direction of weekly trend with volume exhaustion
        long_entry = vol_exhausted and uptrend and at_s3   # fade S3 in uptrend
        short_entry = vol_exhausted and downtrend and at_r3 # fade R3 in downtrend
        
        # Exit logic: price moves back toward mean (daily VWAP approximation) or trend fails
        # Use mid-point of Camarilla width as exit target
        camarilla_width = r3_aligned[i] - s3_aligned[i]
        exit_level_long = s3_aligned[i] + 0.5 * camarilla_width  # 50% retracement
        exit_level_short = r3_aligned[i] - 0.5 * camarilla_width
        
        long_exit = low[i] <= exit_level_long or (not uptrend)
        short_exit = high[i] >= exit_level_short or (not downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_R3_S3_Fade_1wTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0