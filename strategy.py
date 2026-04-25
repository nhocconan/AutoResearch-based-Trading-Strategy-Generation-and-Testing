#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrendFilter_VolumeConfirm_v1
Hypothesis: Trade Donchian(20) breakouts on daily timeframe with 1-week EMA50 trend filter and volume confirmation.
Donchian channels identify volatility-based breakouts; 1-week EMA50 ensures alignment with higher timeframe trend;
volume spike filters false breakouts. Works in both bull and bear markets by capturing momentum bursts.
Target: 7-25 trades/year per symbol (30-100 total over 4 years).
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
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for HTF trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian(20) channels on 1d data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian(20) and EMA50
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF trend (bullish = price above EMA50)
        htf_1w_bullish = close[i] > ema_50_1w_aligned[i]
        htf_1w_bearish = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 0:
            # Long setup: price breaks above Donchian upper + 1w uptrend + volume confirmation
            long_setup = (close[i] > highest_high[i]) and htf_1w_bullish and volume_confirm
            
            # Short setup: price breaks below Donchian lower + 1w downtrend + volume confirmation
            short_setup = (close[i] < lowest_low[i]) and htf_1w_bearish and volume_confirm
            
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
            # Exit: price touches Donchian lower (stop) OR 1w trend turns bearish
            if (close[i] <= lowest_low[i]) or (not htf_1w_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Donchian upper (stop) OR 1w trend turns bullish
            if (close[i] >= highest_high[i]) or (htf_1w_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_1wTrendFilter_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0