#!/usr/bin/env python3
"""
4h Volatility Breakout with Volume Confirmation and Daily Trend Filter.
Trades breakouts above/below 4-hour Donchian channels confirmed by volume spikes,
only when daily price is above/below 50-period EMA to filter range-bound conditions.
Designed for 4h timeframe to target 75-200 total trades over 4 years (19-50/year).
Combines price action, volume confirmation, and trend filtering for robustness.
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4-hour Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Breakout conditions
    breakout_up = high > high_20
    breakout_down = low < low_20
    
    # Daily EMA (50-period) for trend filter
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Trend: price above EMA = uptrend, below EMA = downtrend
    uptrend = df_1d['close'].values > ema_50
    downtrend = df_1d['close'].values < ema_50
    
    # Volume spike: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)
    
    # Align daily signals to 4h timeframe
    uptrend_aligned = align_htf_to_ltf(prices, df_1d, uptrend.astype(float))
    downtrend_aligned = align_htf_to_ltf(prices, df_1d, downtrend.astype(float))
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(uptrend_aligned[i]) or 
            np.isnan(downtrend_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Donchian breakout + volume spike + daily trend
        long_entry = (breakout_up[i] and 
                      vol_spike_aligned[i] > 0.5 and 
                      uptrend_aligned[i] > 0.5)
        short_entry = (breakout_down[i] and 
                       vol_spike_aligned[i] > 0.5 and 
                       downtrend_aligned[i] > 0.5)
        
        # Exit when price returns to middle of Donchian channel
        donchian_mid = (high_20 + low_20) / 2
        exit_long = position == 1 and close[i] <= donchian_mid[i]
        exit_short = position == -1 and close[i] >= donchian_mid[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_volatility_breakout_v1"
timeframe = "4h"
leverage = 1.0