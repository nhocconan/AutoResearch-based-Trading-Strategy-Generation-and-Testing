#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wEMA34_Trend_VolumeSpike
Hypothesis: Daily Donchian(20) breakout with 1-week EMA34 trend filter and volume spike confirmation.
Enters long when price breaks above upper Donchian channel with bullish 1w trend and volume spike.
Enters short when price breaks below lower Donchian channel with bearish 1w trend and volume spike.
Uses discrete position sizing (0.0, ±0.30) to minimize fee churn. Target: 30-100 total trades over 4 years.
Works in both bull and bear markets by following the 1w trend direction only.
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
    
    # Calculate Donchian(20) channels on daily timeframe (prior completed day)
    df_1d = get_htf_data(prices, '1d')
    
    # Prior completed day's high/low for Donchian calculation (shifted by 1)
    prior_high = np.roll(df_1d['high'].values, 1)
    prior_low = np.roll(df_1d['low'].values, 1)
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    
    # Calculate 20-period rolling max/min of prior day's high/low
    high_20 = pd.Series(prior_high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(prior_low).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Load 1w data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.30
    
    # Start after warmup (need 1d shift + 20-period Donchian + 34-period EMA)
    start_idx = 1 + max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above upper Donchian + bullish 1w trend + volume spike
        if close[i] > upper_20_aligned[i] and close[i] > ema_34_1w_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: break below lower Donchian + bearish 1w trend + volume spike
        elif close[i] < lower_20_aligned[i] and close[i] < ema_34_1w_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price reverts to opposite Donchian level
        elif position == 1 and close[i] < lower_20_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > upper_20_aligned[i]:
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

name = "1d_Donchian20_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0