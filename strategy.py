#!/usr/bin/env python3
"""
6h_WeeklyDonchian20_Breakout_1dTrend_VolumeSpike
Hypothesis: On 6h timeframe, breakout above/below weekly Donchian(20) channels with 1d EMA50 trend filter and volume spike confirmation captures sustained moves in both bull and bear markets while minimizing false breakouts. Weekly structure provides robust support/resistance; 6h timeframe reduces noise vs lower timeframes. Targets 12-25 trades/year to stay within fee-efficient range.
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
    
    # Calculate weekly Donchian channels (20-period high/low)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian: upper = rolling max(high, 20), lower = rolling min(low, 20)
    donchian_upper = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 6h timeframe (completed weekly bar only)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for weekly Donchian (20), 1d EMA50 (50), volume MA (30)
    start_idx = max(20, 50, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian upper + 1d uptrend + volume spike
            long_setup = (close[i] > donchian_upper_aligned[i]) and \
                         (close[i] > ema_50_1d_aligned[i]) and \
                         volume_spike[i]
            # Short: price breaks below weekly Donchian lower + 1d downtrend + volume spike
            short_setup = (close[i] < donchian_lower_aligned[i]) and \
                          (close[i] < ema_50_1d_aligned[i]) and \
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
            # Exit: price closes below weekly Donchian lower OR 1d trend turns down
            if (close[i] < donchian_lower_aligned[i]) or \
               (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price closes above weekly Donchian upper OR 1d trend turns up
            if (close[i] > donchian_upper_aligned[i]) or \
               (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyDonchian20_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0