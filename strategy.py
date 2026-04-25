#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian20_Breakout_1dTrend_VolumeConfirm
Hypothesis: 6h Donchian(20) breakout in direction of weekly pivot trend (price > weekly pivot = bullish, < = bearish) with 1d EMA50 trend filter and volume confirmation.
In bull markets: weekly pivot acts as dynamic support, breakouts continue upward.
In bear markets: weekly pivot acts as dynamic resistance, breakouts continue downward.
Volume confirmation filters false breakouts. Discrete sizing (0.25) minimizes fees.
Target: 12-37 trades/year (50-150 over 4 years).
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
    
    # Get 6h data for Donchian calculations (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 21:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Donchian channels for each 6h bar (based on previous 20 bars)
    upper_20 = np.full(len(close_6h), np.nan)
    lower_20 = np.full(len(close_6h), np.nan)
    
    for i in range(20, len(close_6h)):
        high_prev = high_6h[i-20:i]
        low_prev = low_6h[i-20:i]
        upper_20[i] = np.max(high_prev)
        lower_20[i] = np.min(low_prev)
    
    # Align Donchian levels to original timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_6h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_6h, lower_20)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA50 for trend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1w data for weekly pivot (pivot = (H+L+C)/3)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine trend from weekly pivot: price > pivot = bullish, < pivot = bearish
        weekly_bullish = close[i] > weekly_pivot_aligned[i]
        weekly_bearish = close[i] < weekly_pivot_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with weekly bullish bias, 1d EMA50 uptrend, and volume spike
            long_signal = (close[i] > upper_20_aligned[i]) and weekly_bullish and (close[i] > ema_50_1d_aligned[i]) and vol_spike[i]
            # Short: price breaks below lower Donchian with weekly bearish bias, 1d EMA50 downtrend, and volume spike
            short_signal = (close[i] < lower_20_aligned[i]) and weekly_bearish and (close[i] < ema_50_1d_aligned[i]) and vol_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions: price touches lower Donchian or weekly pivot turns bearish
            exit_signal = (close[i] < lower_20_aligned[i]) or (close[i] < weekly_pivot_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: price touches upper Donchian or weekly pivot turns bullish
            exit_signal = (close[i] > upper_20_aligned[i]) or (close[i] > weekly_pivot_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_Donchian20_Breakout_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0