#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_WeeklyTrend_Volume
Hypothesis: Use weekly Camarilla R3/S3 levels as breakout triggers, with weekly EMA34 as trend filter and volume confirmation.
This strategy targets 10-25 trades per year by requiring confluence of weekly structure, trend alignment, and volume spike.
Works in both bull and bear markets by following weekly trend direction. Weekly timeframe reduces noise and false breakouts.
"""

name = "1d_Camarilla_R3_S3_WeeklyTrend_Volume"
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
    
    # Get weekly data for Camarilla levels and trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (R3, S3)
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    weekly_range = df_weekly['high'].values - df_weekly['low'].values
    camarilla_r3 = df_weekly['close'].values + 1.1 * weekly_range / 2
    camarilla_s3 = df_weekly['close'].values - 1.1 * weekly_range / 2
    
    # Align weekly levels to daily timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_weekly, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_weekly, camarilla_s3)
    
    # Weekly EMA34 trend filter
    ema_34_weekly = pd.Series(df_weekly['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_34_weekly)
    
    # Daily volume filter: current volume > 2x 20-day average
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2 * avg_volume_20)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for weekly indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_weekly_aligned[i]) or np.isnan(volume_filter[i]) or
            np.isnan(session_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R3 + weekly uptrend + volume spike + session
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema_34_weekly_aligned[i] and volume_filter[i] and session_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S3 + weekly downtrend + volume spike + session
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema_34_weekly_aligned[i] and volume_filter[i] and session_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below weekly EMA34 or weekly S3 level
            if close[i] < ema_34_weekly_aligned[i] or close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above weekly EMA34 or weekly R3 level
            if close[i] > ema_34_weekly_aligned[i] or close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals