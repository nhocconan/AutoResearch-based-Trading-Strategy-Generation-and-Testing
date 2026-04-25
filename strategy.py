#!/usr/bin/env python3
"""
1d_Donchian20_1wTrend_Filter_VolumeSpike
Hypothesis: Daily Donchian(20) breakout with 1-week trend filter and volume confirmation.
Enters long when price breaks above 20-day high with 1w uptrend (close > 1w EMA50) and volume spike (>2.0x 20-day avg volume).
Enters short when price breaks below 20-day low with 1w downtrend (close < 1w EMA50) and volume spike.
Uses discrete position sizing (0.25) to limit fee drag. Designed for 1d timeframe with ~10-25 trades/year.
Works in both bull and bear markets via trend filter that adapts to 1-week momentum.
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
    
    # 1-week data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1-week EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily data for Donchian channels (20-day high/low)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels: 20-day high/low
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (no additional delay needed for breakout)
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Volume confirmation: current volume > 2.0x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 and Donchian channels
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 20-day high with 1w uptrend and volume spike
            long_setup = (close[i] > high_20_aligned[i]) and (close[i] > ema_50_1w_aligned[i]) and volume_spike[i]
            # Short: price breaks below 20-day low with 1w downtrend and volume spike
            short_setup = (close[i] < low_20_aligned[i]) and (close[i] < ema_50_1w_aligned[i]) and volume_spike[i]
            
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
            # Exit: price breaks below 20-day low OR 1w trend turns down
            if (close[i] < low_20_aligned[i]) or (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above 20-day high OR 1w trend turns up
            if (close[i] > high_20_aligned[i]) or (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_1wTrend_Filter_VolumeSpike"
timeframe = "1d"
leverage = 1.0