#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike
Hypothesis: Trade 12h Donchian(20) breakouts only when 1w EMA50 confirms trend (price above/below EMA) and volume spikes (>2.0x 20-bar MA). Donchian channels provide robust trend-following structure, while 1w EMA filters for primary trend alignment. Volume spike confirms institutional participation. Uses discrete sizing 0.25 to limit fee drag. Target 12-37 trades/year on 12h timeframe.
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 12h (completed 1w bar only)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20), EMA50 (1w), volume MA (20)
    start_idx = max(20, 50)  # 50 for EMA warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Donchian channels for current bar (using data up to i)
        lookback_high = np.max(high[i-19:i+1]) if i >= 19 else np.nan
        lookback_low = np.min(low[i-19:i+1]) if i >= 19 else np.nan
        
        if position == 0:
            # Long: price breaks above Donchian upper + above 1w EMA50 + volume spike
            long_setup = (close[i] > lookback_high) and \
                         (close[i] > ema_50_1w_aligned[i]) and \
                         volume_spike[i]
            # Short: price breaks below Donchian lower + below 1w EMA50 + volume spike
            short_setup = (close[i] < lookback_low) and \
                          (close[i] < ema_50_1w_aligned[i]) and \
                          volume_spike[i]
            
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
            # Exit: price closes below Donchian lower OR below 1w EMA50
            if (close[i] < lookback_low) or \
               (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price closes above Donchian upper OR above 1w EMA50
            if (close[i] > lookback_high) or \
               (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0