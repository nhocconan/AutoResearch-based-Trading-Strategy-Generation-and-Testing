#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Trade 12h Donchian(20) breakouts with 1d EMA50 trend filter and volume confirmation.
- Trend filter: price > 1d EMA50 = bullish, price < 1d EMA50 = bearish.
- In bullish 1d trend: buy breakouts above upper Donchian(20), sell breakdowns below lower Donchian(20).
- In bearish 1d trend: sell breakdowns below lower Donchian(20), buy breakouts above upper Donchian(20) (continuation logic).
- Volume confirmation: require volume > 2.0x 20-period average to avoid false breakouts.
- Exit on trend reversal or mean reversion to midpoint of Donchian channel.
- Position size: 0.25. Target: 50-150 total trades over 4 years = 12-37/year.
- Works in both bull and bear: 1d trend filter captures major moves, volume filter reduces noise.
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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h Donchian(20) channels
    high_12h = get_htf_data(prices, '12h')['high'].values
    low_12h = get_htf_data(prices, '12h')['low'].values
    
    # Upper channel: max(high, 20)
    upper_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower channel: min(low, 20)
    lower_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    # Midpoint: (upper + lower) / 2
    midpoint_12h = (upper_12h + lower_12h) / 2.0
    
    # Align Donchian levels to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '12h'), upper_12h)
    lower_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '12h'), lower_12h)
    midpoint_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '12h'), midpoint_12h)
    
    # Volume spike confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA(50), Donchian(20), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(upper_aligned[i]) or
            np.isnan(lower_aligned[i]) or
            np.isnan(midpoint_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend using EMA50
        htf_1d_bullish = close[i] > ema_50_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Breakout logic: trade in direction of 1d trend with volume confirmation
            long_setup = (close[i] > upper_aligned[i]) and htf_1d_bullish and volume_spike[i]
            short_setup = (close[i] < lower_aligned[i]) and htf_1d_bearish and volume_spike[i]
            
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
            # Exit on trend reversal or mean reversion to midpoint
            exit_signal = (not htf_1d_bullish) or (close[i] < midpoint_aligned[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit on trend reversal or mean reversion to midpoint
            exit_signal = htf_1d_bullish or (close[i] > midpoint_aligned[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0