#!/usr/bin/env python3
# 4h_Donchian20_1dTrend_VolumeSpike
# Hypothesis: 4h Donchian(20) breakout in direction of 1d trend (EMA50), confirmed by volume spike (>1.5x 20-period average).
# Exits when price crosses the opposite Donchian boundary (10-period) or volume drops below average.
# Designed to work in both bull and bear markets by following the higher timeframe trend.
# Targets ~25-40 trades per year on 4h timeframe with position size 0.25.

name = "4h_Donchian20_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for trend (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period for entry, 10-period for exit)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.max(arr[i - window + 1:i + 1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.min(arr[i - window + 1:i + 1])
        return res
    
    donchian_high_20 = rolling_max(high, 20)
    donchian_low_20 = rolling_min(low, 20)
    donchian_high_10 = rolling_max(high, 10)
    donchian_low_10 = rolling_min(low, 10)
    
    # Calculate volume average (20-period)
    vol_avg_20 = np.zeros_like(volume)
    for i in range(19, len(volume)):
        vol_avg_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)  # Warmup for Donchian(20) and volume average
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-period average
        vol_spike = volume[i] > 1.5 * vol_avg_20[i]
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian(20) high + volume spike + uptrend
            if (close[i] > donchian_high_20[i] and 
                vol_spike and 
                price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian(20) low + volume spike + downtrend
            elif (close[i] < donchian_low_20[i] and 
                  vol_spike and 
                  price_below_ema):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below Donchian(10) low OR volume drops below average
            if (close[i] < donchian_low_10[i] or 
                volume[i] < vol_avg_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above Donchian(10) high OR volume drops below average
            if (close[i] > donchian_high_10[i] or 
                volume[i] < vol_avg_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals