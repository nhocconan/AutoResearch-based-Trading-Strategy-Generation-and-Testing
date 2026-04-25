#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrendFilter_VolumeSpike_v1
Hypothesis: Trade 12h Donchian(20) breakouts with 1d EMA50 trend filter and volume spike confirmation.
Donchian channels provide objective breakout levels that work in both trending and ranging markets.
In strong 1d trends (price above/below EMA50), breakouts in the trend direction have higher follow-through.
Volume spike confirms institutional participation. Discrete sizing (0.25) limits fee drift.
Target: 12-37 trades/year per symbol to survive changing regimes from 2021-2026.
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
    
    # Get 1d data for HTF trend filter and Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) from 1d data (using previous 20 days for breakout)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper band: highest high of previous 20 days
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
    # Lower band: lowest low of previous 20 days
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume spike: current 12h volume > 2.0 * 20-period 12h volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20+1) and EMA50 (50) and volume MA (20)
    start_idx = max(50, 21, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above EMA50)
        htf_1d_bullish = close[i] > ema_50_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above Donchian high + 1d uptrend + volume spike
            long_setup = (close[i] > donchian_high_aligned[i]) and htf_1d_bullish and volume_spike[i]
            
            # Short setup: price breaks below Donchian low + 1d downtrend + volume spike
            short_setup = (close[i] < donchian_low_aligned[i]) and htf_1d_bearish and volume_spike[i]
            
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
            # Exit: price touches Donchian low (stop) OR 1d trend turns bearish
            if (close[i] <= donchian_low_aligned[i]) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Donchian high (stop) OR 1d trend turns bullish
            if (close[i] >= donchian_high_aligned[i]) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_Breakout_1dTrendFilter_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0