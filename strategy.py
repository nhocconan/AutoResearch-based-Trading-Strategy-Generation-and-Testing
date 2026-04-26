#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume spike (>2.0x median).
Enters long when price breaks above upper Donchian(20) with volume spike and bullish 12h trend (price > EMA50).
Enters short when price breaks below lower Donchian(20) with volume spike and bearish 12h trend (price < EMA50).
Exits on opposite Donchian breakout.
Uses discrete position sizing (0.25) to minimize churn. Target: 75-200 trades over 4 years.
Works in both bull and bear markets by following 12h trend filter.
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
    
    # Calculate Donchian channels for 4h (based on previous 20 bars)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_donchian = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    lower_donchian = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Load 12h data for HTF trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 2.0x 50-period median
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=50, min_periods=50).median().values
    volume_spike = volume > (2.0 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20-period Donchian, 50-period EMA, 50-period volume median)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_median[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: price breaks above upper Donchian + volume spike + bullish 12h trend
        if close[i] > upper_donchian[i] and volume_spike[i] and close[i] > ema_50_12h_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below lower Donchian + volume spike + bearish 12h trend
        elif close[i] < lower_donchian[i] and volume_spike[i] and close[i] < ema_50_12h_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: opposite Donchian breakout
        elif position == 1 and close[i] < lower_donchian[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > upper_donchian[i]:
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

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0