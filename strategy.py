#!/usr/bin/env python3
"""
6h_Donchian_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: On 6h timeframe, enter long when price breaks above 20-bar Donchian high AND 1d trend is up (EMA50) AND volume > 1.5x 20-bar average volume; enter short when price breaks below 20-bar Donchian low AND 1d trend is down AND volume > 1.5x average volume. Uses Donchian for structure, 1d EMA50 for trend filter, and volume confirmation to avoid false breakouts. Targets 12-37 trades per year with discrete sizing (0.0, ±0.25) to minimize fee drag. Works in bull via trend continuation breakouts and in bear via breakdowns with volume confirmation.
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
    
    # Get 6h data for Donchian calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels on 6h data
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # 20-period Donchian high and low
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low)
    
    # Get 1d data for HTF trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period average volume on 6h
    volume_6h = df_6h['volume'].values
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    avg_volume_20_aligned = align_htf_to_ltf(prices, df_6h, avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian (20) + EMA (50) + volume avg (20)
    start_idx = max(20, 50, 20)  # EMA50 needs 50 periods
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(avg_volume_20_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high_aligned[i]
        breakout_down = close[i] < donchian_low_aligned[i]
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * avg_volume_20_aligned[i]
        
        # 1d trend filter
        trend_uptrend = close[i] > ema_50_1d_aligned[i]
        trend_downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: breakout up + volume confirmation + 1d uptrend
            long_signal = breakout_up and volume_confirmed and trend_uptrend
            
            # Short: breakout down + volume confirmation + 1d downtrend
            short_signal = breakout_down and volume_confirmed and trend_downtrend
            
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
            # Exit: breakout down OR trend change to downtrend
            if breakout_down or not trend_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: breakout up OR trend change to uptrend
            if breakout_up or not trend_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0