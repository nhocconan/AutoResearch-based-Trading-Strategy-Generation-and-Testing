#!/usr/bin/env python3
"""
1d_Donchian20_WeeklyTrend_VolumeSpike
Hypothesis: Daily Donchian(20) breakout with weekly EMA50 trend filter and volume spike confirmation.
Enters long when price breaks above upper Donchian channel with bullish weekly trend and volume spike.
Enters short when price breaks below lower Donchian channel with bearish weekly trend and volume spike.
Uses discrete position sizing (0.0, ±0.30) to minimize fee churn. Designed for 30-100 total trades over 4 years.
Works in both bull and bear markets by following the weekly trend direction only.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period) on daily timeframe
    # Use prior completed daily bar to avoid look-ahead
    df_1d = get_htf_data(prices, '1d')
    
    # Prior day's high/low for Donchian calculation (shifted by 1)
    prior_high = np.roll(df_1d['high'].values, 1)
    prior_low = np.roll(df_1d['low'].values, 1)
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    
    # 20-period rolling max/min on prior daily data
    high_series = pd.Series(prior_high)
    low_series = pd.Series(prior_low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA50 trend
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.30
    
    # Start after warmup (need 20-day Donchian + 50-week EMA + 1-day shift)
    start_idx = 20 + 50 + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above upper Donchian + bullish weekly trend + volume spike
        if close[i] > donchian_high_aligned[i] and close[i] > ema_50_1w_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: break below lower Donchian + bearish weekly trend + volume spike
        elif close[i] < donchian_low_aligned[i] and close[i] < ema_50_1w_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price reverts to opposite Donchian level
        elif position == 1 and close[i] < donchian_low_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > donchian_high_aligned[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_Donchian20_WeeklyTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0