#!/usr/bin/env python3
"""
6h_Weekly_Pivot_PB_Donchian20_1dEMA50_Trend
Hypothesis: On 6h timeframe, enter long when price pulls back to weekly pivot point after breaking above weekly Donchian high AND 1d EMA50 uptrend AND volume > 1.5x 20-period average. Enter short when price pulls back to weekly pivot after breaking below weekly Donchian low AND 1d EMA50 downtrend AND volume spike. This combines weekly structure (pivot/Donchian) with daily trend filter and volume confirmation for high-probability retracement entries in both bull and bear markets. Targets 12-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot and Donchian
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot point (PP) from previous weekly bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    prev_close_1w[0] = np.nan
    
    weekly_pivot = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    
    # Calculate weekly Donchian channels (20-period) from previous weekly bar
    donchian_high = pd.Series(prev_high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(prev_low_1w).rolling(window=20, min_periods=20).min().values
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Align weekly levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly Donchian warmup (20) and volume MA warmup (20)
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Pullback to weekly pivot conditions
        near_pivot = abs(close[i] - weekly_pivot_aligned[i]) < (donchian_high_aligned[i] - donchian_low_aligned[i]) * 0.02
        
        # Weekly Donchian breakout conditions (must have broken in recent past)
        broke_above_donchian = np.any(high[max(0, i-20):i+1] > donchian_high_aligned[i]) if i >= 20 else False
        broke_below_donchian = np.any(low[max(0, i-20):i+1] < donchian_low_aligned[i]) if i >= 20 else False
        
        # 1d trend filter
        trend_uptrend = close[i] > ema_50_1d_aligned[i]
        trend_downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: pullback to weekly pivot after breaking above weekly Donchian + volume spike + 1d uptrend
            long_signal = near_pivot and broke_above_donchian and volume_spike[i] and trend_uptrend
            
            # Short: pullback to weekly pivot after breaking below weekly Donchian + volume spike + 1d downtrend
            short_signal = near_pivot and broke_below_donchian and volume_spike[i] and trend_downtrend
            
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
            # Exit: price moves above weekly pivot (failure) or trend change
            if close[i] > weekly_pivot_aligned[i] * 1.01 or not trend_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price moves below weekly pivot (failure) or trend change
            if close[i] < weekly_pivot_aligned[i] * 0.99 or not trend_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Weekly_Pivot_PB_Donchian20_1dEMA50_Trend"
timeframe = "6h"
leverage = 1.0