#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_Direction_VolumeConfirm_v1
Hypothesis: Trade 6h Donchian(20) breakouts aligned with weekly Camarilla pivot direction (from R3/S3 levels) and volume confirmation. Weekly pivot provides structural bias; Donchian breakout captures momentum; volume filters false signals. Works in bull/bear via weekly trend filter. Target 12-35 trades/year (50-150 over 4 years). Discrete size 0.25 to limit fee drag.
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
    
    # Get weekly data for Camarilla pivot levels (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # need sufficient weekly data
        return np.zeros(n)
    
    # Calculate weekly Camarilla R3 and S3 from previous weekly bar
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w[0] = np.nan  # first bar has no previous
    
    # Weekly Camarilla R3 and S3 (using previous weekly bar OHLC)
    camarilla_r3_1w = prev_close_1w + 1.1 * (prev_high_1w - prev_low_1w) * 1.1 / 4
    camarilla_s3_1w = prev_close_1w - 1.1 * (prev_high_1w - prev_low_1w) * 1.1 / 4
    
    # Weekly trend: price above/below weekly Camarilla midpoint
    weekly_midpoint = (camarilla_r3_1w + camarilla_s3_1w) / 2
    weekly_trend_up = close_1w > weekly_midpoint
    weekly_trend_down = close_1w < weekly_midpoint
    
    # Align weekly data to 6h timeframe
    camarilla_r3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3_1w)
    camarilla_s3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3_1w)
    weekly_midpoint_aligned = align_htf_to_ltf(prices, df_1w, weekly_midpoint)
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up.astype(float))
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down.astype(float))
    
    # 6h Donchian(20) breakout levels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.8x 30-period average on 6h
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=30, min_periods=30).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.8
    
    # Require close to stay beyond Donchian level for 2 consecutive bars to reduce false breakouts
    close_above_donchian_high = close > donchian_high
    close_below_donchian_low = close < donchian_low
    close_above_donchian_high_2bar = close_above_donchian_high & np.roll(close_above_donchian_high, 1)
    close_below_donchian_low_2bar = close_below_donchian_low & np.roll(close_below_donchian_low, 1)
    close_above_donchian_high_2bar[0] = False
    close_below_donchian_low_2bar[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian (20), volume MA (30)
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_1w_aligned[i]) or np.isnan(camarilla_s3_1w_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Weekly trend alignment
        trend_up = weekly_trend_up_aligned[i] > 0.5
        trend_down = weekly_trend_down_aligned[i] > 0.5
        
        if position == 0:
            # Long: price breaks above Donchian high + volume spike + weekly uptrend + 2-bar confirmation
            long_breakout = close_above_donchian_high_2bar[i]
            long_signal = long_breakout and volume_spike[i] and trend_up
            
            # Short: price breaks below Donchian low + volume spike + weekly downtrend + 2-bar confirmation
            short_breakout = close_below_donchian_low_2bar[i]
            short_signal = short_breakout and volume_spike[i] and trend_down
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price touches Donchian low OR weekly trend turns down
            if (close[i] < donchian_low[i] or not trend_up):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price touches Donchian high OR weekly trend turns up
            if (close[i] > donchian_high[i] or not trend_down):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Direction_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0