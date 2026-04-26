#!/usr/bin/env python3
"""
1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeFilter_Session
Hypothesis: Trade 1h breakouts from 4h Camarilla R3/S3 levels with 4h EMA50 trend filter and volume confirmation. Session filter (08-20 UTC) reduces noise. Uses 4h for signal direction (trend + Camarilla), 1h only for entry timing precision. Discrete size 0.20 limits fee drag. Target 15-37 trades/year.
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla pivot levels and trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Camarilla pivot levels (R3, S3) from previous 4h bar
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    prev_close = np.roll(close_4h, 1)
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close[0] = np.nan
    
    # Calculate Camarilla R3 and S3 for previous 4h bar
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 4
    
    # 4h EMA50 trend filter
    close_4h_series = pd.Series(close_4h)
    ema_50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 4h data to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: volume > 1.5x 20-period average on 1h
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.5
    
    # Require close to stay beyond level for 2 consecutive bars to reduce false breakouts
    close_above_r3 = close > camarilla_r3_aligned
    close_below_s3 = close < camarilla_s3_aligned
    close_above_r3_2bar = close_above_r3 & np.roll(close_above_r3, 1)
    close_below_s3_2bar = close_below_s3 & np.roll(close_below_s3, 1)
    close_above_r3_2bar[0] = False
    close_below_s3_2bar[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50 (50), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Skip if outside session
        if not in_session[i]:
            # Hold current position outside session
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # 4h trend alignment
        trend_4h_uptrend = close[i] > ema_50_4h_aligned[i]
        trend_4h_downtrend = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 + volume spike + 4h uptrend + 2-bar confirmation
            long_breakout = close_above_r3_2bar[i]
            long_signal = long_breakout and volume_spike[i] and trend_4h_uptrend
            
            # Short: price breaks below S3 + volume spike + 4h downtrend + 2-bar confirmation
            short_breakout = close_below_s3_2bar[i]
            short_signal = short_breakout and volume_spike[i] and trend_4h_downtrend
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: price touches S3 level OR 4h trend turns down
            if (close[i] < camarilla_s3_aligned[i] or not trend_4h_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: price touches R3 level OR 4h trend turns up
            if (close[i] > camarilla_r3_aligned[i] or not trend_4h_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeFilter_Session"
timeframe = "1h"
leverage = 1.0