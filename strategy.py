#!/usr/bin/env python3
"""
1d_WeeklyDonchian20_Breakout_WeeklyTrend
Hypothesis: Daily Donchian(20) breakout with weekly EMA34 trend filter and volume confirmation.
Works in bull markets (breakouts with weekly uptrend) and bear markets (fades from weekly extremes with volume).
Designed for low trade frequency (~10-25/year) to minimize fee drag on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily Donchian(20) channels
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 1d timeframe (completed daily bar only)
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20 bars), EMA34 (34), volume MA (20)
    start_idx = max(20, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price closes above upper Donchian + weekly uptrend + volume spike
            long_setup = (close[i] > high_20_aligned[i]) and \
                         (close[i] > ema_34_1w_aligned[i]) and \
                         volume_spike[i]
            # Short: price closes below lower Donchian + weekly downtrend + volume spike
            short_setup = (close[i] < low_20_aligned[i]) and \
                          (close[i] < ema_34_1w_aligned[i]) and \
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
            # Exit: price closes below lower Donchian OR weekly trend turns down
            if (close[i] < low_20_aligned[i]) or \
               (close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price closes above upper Donchian OR weekly trend turns up
            if (close[i] > high_20_aligned[i]) or \
               (close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WeeklyDonchian20_Breakout_WeeklyTrend"
timeframe = "1d"
leverage = 1.0