#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d trend filter and volume confirmation.
# Breakout above 20-period high in 1d uptrend = long, breakdown below 20-period low in 1d downtrend = short.
# Uses 1d EMA(50) for trend filter and 20-period volume spike for confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "12h_Donchian20_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up = ema_50_1d[1:] > ema_50_1d[:-1]  # Rising EMA = uptrend
    trend_up = np.concatenate([[False], trend_up])  # Align with 1d index
    
    # 20-period Donchian channels on 12h data
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period volume spike (1.8x EMA)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.8)
    
    # Align 1d indicators to 12h timeframe
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or
            np.isnan(trend_up_aligned[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: breakout above 20-period high in uptrend
            if (trend_up_aligned[i] > 0.5 and  # 1d uptrend
                close[i] >= high_max[i] and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: breakdown below 20-period low in downtrend
            elif (trend_up_aligned[i] <= 0.5 and  # 1d downtrend
                  close[i] <= low_min[i] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: reverse signal or breakdown below 20-period low
            if (trend_up_aligned[i] <= 0.5 and  # 1d downtrend
                close[i] <= low_min[i]):  # Break below 20-period low
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: reverse signal or breakout above 20-period high
            if (trend_up_aligned[i] > 0.5 and  # 1d uptrend
                  close[i] >= high_max[i]):  # Break above 20-period high
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals