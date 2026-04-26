#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian_Breakout_v2
Hypothesis: Trade 6h Donchian(20) breakouts with weekly pivot direction filter and volume confirmation.
Weekly pivot acts as structural support/resistance; price above weekly PP = bull bias, below = bear bias.
Donchian breakout in direction of weekly trend captures momentum with reduced false signals.
Volume confirmation ensures breakouts have participation. Targets 50-150 trades over 4 years.
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
    
    # Get weekly data for pivot calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (PP, R1, S1) from previous weekly bar
    # PP = (H+L+C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    # Avoid NaN from shift
    prev_high = np.where(np.isnan(prev_high), df_1w['high'].values, prev_high)
    prev_low = np.where(np.isnan(prev_low), df_1w['low'].values, prev_low)
    prev_close = np.where(np.isnan(prev_close), df_1w['close'].values, prev_close)
    
    weekly_pp = (prev_high + prev_low + prev_close) / 3.0
    weekly_r1 = 2 * weekly_pp - prev_low
    weekly_s1 = 2 * weekly_pp - prev_high
    
    # Align weekly levels to 6h
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Determine weekly trend: price above/below weekly PP
    weekly_trend_up = close > weekly_pp_aligned  # bullish bias
    weekly_trend_down = close < weekly_pp_aligned  # bearish bias
    
    # Get daily data for Donchian(20) calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate Donchian(20) channels from previous 20 daily bars
    # Upper = max(high of past 20 days), Lower = min(low of past 20 days)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 6h
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Volume confirmation: current volume > 1.5 * 24-period average (4d average on 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of weekly pivot calculation, Donchian(20), volume MA(24)
    start_idx = max(20, 24) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_pp_aligned[i]) or
            np.isnan(donchian_upper_aligned[i]) or
            np.isnan(donchian_lower_aligned[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND volume confirm AND weekly bullish bias
            long_signal = (close_val > donchian_upper_aligned[i]) and vol_conf and weekly_trend_up[i]
            
            # Short: price breaks below Donchian lower AND volume confirm AND weekly bearish bias
            short_signal = (close_val < donchian_lower_aligned[i]) and vol_conf and weekly_trend_down[i]
            
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
            # Exit: price drops below Donchian lower (failed breakout) OR weekly trend flips bearish
            if (close_val < donchian_lower_aligned[i]) or (not weekly_trend_up[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above Donchian upper (failed breakdown) OR weekly trend flips bullish
            if (close_val > donchian_upper_aligned[i]) or (not weekly_trend_down[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_Donchian_Breakout_v2"
timeframe = "6h"
leverage = 1.0