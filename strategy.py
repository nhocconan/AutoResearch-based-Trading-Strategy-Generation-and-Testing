#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrendFilter_VolumeSpike
Hypothesis: Trade daily Donchian(20) breakouts with 1-week EMA50 trend filter and volume spike confirmation.
Uses discrete sizing (0.25) to limit fee drag. Designed for both bull and bear markets by aligning with 1-week trend.
Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for HTF trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period) from daily data
    # Using rolling window on daily close prices
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20) and volume MA (20)
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(high_max[i]) or np.isnan(low_min[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF trend (bullish = price above EMA50)
        htf_1w_bullish = close[i] > ema_50_1w_aligned[i]
        htf_1w_bearish = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above upper Donchian + 1w uptrend + volume confirmation
            long_setup = (close[i] > high_max[i]) and htf_1w_bullish and volume_confirm[i]
            
            # Short setup: price breaks below lower Donchian + 1w downtrend + volume confirmation
            short_setup = (close[i] < low_min[i]) and htf_1w_bearish and volume_confirm[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches lower Donchian (stop) OR 1w trend turns bearish
            if (close[i] <= low_min[i]) or (not htf_1w_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches upper Donchian (stop) OR 1w trend turns bullish
            if (close[i] >= high_max[i]) or (htf_1w_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_1wTrendFilter_VolumeSpike"
timeframe = "1d"
leverage = 1.0