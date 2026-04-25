#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike
Hypothesis: Daily Donchian(20) breakout with 1-week EMA50 trend filter and volume confirmation. Uses HTF 1w for trend alignment (price > 1w EMA50 for long, < 1w EMA50 for short) to reduce whipsaw and capture major trends. Volume confirmation requires >2.0x 20-day mean volume. Designed for low trade frequency (7-25/year) to minimize fee drag while capturing strong directional moves in both bull and bear markets via breakouts with volume and trend alignment.
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
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on 1w for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to daily timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian(20) channels from daily data
    # Upper channel: 20-period high
    # Lower channel: 20-period low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0x 20-day mean volume
    vol_mean_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_mean_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian and EMA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or
            np.isnan(vol_mean_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper channel in uptrend (price > 1w EMA50) with volume confirmation
            # Short: price breaks below Donchian lower channel in downtrend (price < 1w EMA50) with volume confirmation
            long_signal = (close[i] > donchian_upper[i]) and (close[i] > ema_50_1w_aligned[i]) and vol_confirm[i]
            short_signal = (close[i] < donchian_lower[i]) and (close[i] < ema_50_1w_aligned[i]) and vol_confirm[i]
            
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
            # Exit when price moves back below Donchian lower channel (breakout failure)
            exit_signal = close[i] < donchian_lower[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Donchian upper channel (breakout failure)
            exit_signal = close[i] > donchian_upper[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0