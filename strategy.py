#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrendFilter_VolumeSpike_v1
Hypothesis: Trade Donchian(20) breakouts on 12h with 1d EMA50 trend filter and volume spike confirmation.
Long: Price breaks above upper Donchian(20) + price > 1d EMA50 + volume > 2.0 * 20-period avg volume.
Short: Price breaks below lower Donchian(20) + price < 1d EMA50 + volume > 2.0 * 20-period avg volume.
Exit: Opposite Donchian level touch OR trend reversal.
Position size: 0.25 (25% of capital) to limit fee drag and manage drawdown.
Target: 12-37 trades/year (50-150 total over 4 years) to stay within proven winning range for 12h.
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 12h data
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 12h volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20), EMA50 (50), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(high_max[i]) or np.isnan(low_min[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above EMA50)
        htf_1d_bullish = close[i] > ema_50_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above upper Donchian + 1d uptrend + volume spike
            long_setup = (close[i] > high_max[i]) and htf_1d_bullish and volume_spike[i]
            
            # Short setup: price breaks below lower Donchian + 1d downtrend + volume spike
            short_setup = (close[i] < low_min[i]) and htf_1d_bearish and volume_spike[i]
            
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
            # Exit: price touches lower Donchian (stop) OR 1d trend turns bearish
            if (close[i] <= low_min[i]) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches upper Donchian (stop) OR 1d trend turns bullish
            if (close[i] >= high_max[i]) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_Breakout_1dTrendFilter_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0