#!/usr/bin/env python3
# 4h_1d_Donchian20_Breakout_VolumeTrend
# Hypothesis: On 4h timeframe, trade Donchian(20) breakouts with volume spike confirmation and 1d EMA trend filter.
# Donchian(20) provides clear breakout levels, volume confirms institutional interest, and 1d EMA ensures alignment with higher timeframe trend.
# Works in both bull and bear markets by following the 1d trend direction (long only in uptrend, short only in downtrend).
# Targets 20-50 trades per year by requiring strong breakouts with volume confirmation.

name = "4h_1d_Donchian20_Breakout_VolumeTrend"
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA20 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_20_1d = close_1d_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d EMA to 4h timeframe
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema_20_1d_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high, volume spike, and price above 1d EMA20 (uptrend)
            if (close[i] > donchian_high[i] and 
                volume[i] > 2.0 * volume_ma[i] and
                close[i] > ema_20_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, volume spike, and price below 1d EMA20 (downtrend)
            elif (close[i] < donchian_low[i] and 
                  volume[i] > 2.0 * volume_ma[i] and
                  close[i] < ema_20_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or trend reversal (below EMA20)
            if close[i] < donchian_low[i] or close[i] < ema_20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or trend reversal (above EMA20)
            if close[i] > donchian_high[i] or close[i] > ema_20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals