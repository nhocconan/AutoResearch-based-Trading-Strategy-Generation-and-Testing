#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian_Breakout_1dTrend_VolumeSpike
Hypothesis: Trade 6h Donchian(20) breakouts in direction of 1d trend with weekly pivot confirmation and volume spike.
Weekly pivots provide strong structural support/resistance that works across market regimes.
Donchian breakouts capture momentum, filtered by 1d EMA50 trend and volume spikes to avoid false signals.
Designed for low turnover (target: 50-150 trades over 4 years) to minimize fee drag in 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter and Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for weekly pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) channels on 6h
    # Upper = max(high, 20), Lower = min(low, 20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike filter: volume > 2.0 * 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Standard formula: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    # We'll use the weekly pivot as trend filter: price > P = bullish bias
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    # Avoid NaN from shift
    prev_week_high = np.where(np.isnan(prev_week_high), df_1w['high'].values, prev_week_high)
    prev_week_low = np.where(np.isnan(prev_week_low), df_1w['low'].values, prev_week_low)
    prev_week_close = np.where(np.isnan(prev_week_close), df_1w['close'].values, prev_week_close)
    
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1d EMA(50), Donchian(20), volume MA(50)
    start_idx = max(50, 20, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or
            np.isnan(weekly_pivot_aligned[i]) or
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
        trend_1d_up = close_val > ema_50_1d_aligned[i]   # 1d uptrend
        trend_1d_down = close_val < ema_50_1d_aligned[i]  # 1d downtrend
        above_weekly_pivot = close_val > weekly_pivot_aligned[i]
        below_weekly_pivot = close_val < weekly_pivot_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND 1d trend up AND above weekly pivot AND volume spike
            long_signal = (close_val > donchian_upper[i]) and trend_1d_up and above_weekly_pivot and vol_spike
            
            # Short: price breaks below Donchian lower AND 1d trend down AND below weekly pivot AND volume spike
            short_signal = (close_val < donchian_lower[i]) and trend_1d_down and below_weekly_pivot and vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: 1d trend flips down OR price breaks below weekly pivot (failed breakout)
            if (not trend_1d_up) or (close_val < weekly_pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: 1d trend flips up OR price breaks above weekly pivot (failed breakdown)
            if (not trend_1d_down) or (close_val > weekly_pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_Donchian_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0