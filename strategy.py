#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h 1-week Donchian breakout with 1-day volume confirmation
# Uses 1-week Donchian(20) channels to capture long-term breakouts in both bull and bear markets
# Confirmed by 1-day volume > 2x average to ensure breakout strength
# Low turnover expected: ~15-30 trades/year per symbol to avoid fee drag
# Works in bull markets (breakouts up) and bear markets (breakouts down)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-week data ONCE for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1-week Donchian channels (20 periods)
    donch_len = 20
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    upper_channel = pd.Series(high_1w).rolling(window=donch_len, min_periods=donch_len).max().values
    lower_channel = pd.Series(low_1w).rolling(window=donch_len, min_periods=donch_len).min().values
    
    # Align 1-week Donchian channels to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower_channel)
    
    # Load 1-day data ONCE for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day volume average (20 periods)
    volume_1d = df_1d['volume'].values
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1-day volume average to 12h timeframe
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(100, donch_len * 3, 20)  # Ensure sufficient warmup
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2x average
        volume_confirmed = volume[i] > 2.0 * vol_ma_aligned[i]
        
        if position == 0:
            # Enter long: 1-week Donchian breakout up + volume confirmation
            if (close[i] > upper_aligned[i-1] and volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: 1-week Donchian breakout down + volume confirmation
            elif (close[i] < lower_aligned[i-1] and volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 1-week lower channel
            if close[i] < lower_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above 1-week upper channel
            if close[i] > upper_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1wDonchian_1dVol_Breakout_v1"
timeframe = "12h"
leverage = 1.0