#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Use weekly trend filter (price above/below weekly 50 EMA) to avoid counter-trend trades.
Enter on Camarilla R3/S3 breakout with volume spike (2x 20-period average) in direction of weekly trend.
Exit on opposite Camarilla level (R4/S4) or when volume drops below average.
Designed for 6h timeframe to target 12-37 trades/year with controlled risk.
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
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly 50 EMA for trend filter
    close_weekly = df_weekly['close'].values
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA50 to 6h timeframe
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Weekly trend: bullish when close > EMA50
    weekly_uptrend = close_weekly > ema50_weekly
    weekly_downtrend = close_weekly < ema50_weekly
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_weekly, weekly_uptrend.astype(float)) > 0.5
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_weekly, weekly_downtrend.astype(float)) > 0.5
    
    # Get daily data for Camarilla levels
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_daily['high'].shift(1).values
    low_prev = df_daily['low'].shift(1).values
    close_prev = df_daily['close'].shift(1).values
    
    # Camarilla formulas
    range_prev = high_prev - low_prev
    camarilla_r3 = close_prev + range_prev * 1.1 / 6
    camarilla_s3 = close_prev - range_prev * 1.1 / 6
    camarilla_r4 = close_prev + range_prev * 1.1 / 2
    camarilla_s4 = close_prev - range_prev * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s3)
    r4_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s4)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for weekly EMA50 to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_weekly_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > r3_aligned[i]
        short_breakout = close[i] < s3_aligned[i]
        
        # Exit conditions (opposite Camarilla levels)
        long_exit = close[i] > r4_aligned[i]
        short_exit = close[i] < s4_aligned[i]
        
        # Entry logic with weekly trend alignment and volume spike
        if (weekly_uptrend_aligned[i] and long_breakout[i] and volume_spike[i] and position <= 0):
            signals[i] = 0.25
            position = 1
        elif (weekly_downtrend_aligned[i] and short_breakout[i] and volume_spike[i] and position >= 0):
            signals[i] = -0.25
            position = -1
        # Exit logic
        elif (long_exit[i] and position == 1):
            signals[i] = 0.0
            position = 0
        elif (short_exit[i] and position == -1):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0