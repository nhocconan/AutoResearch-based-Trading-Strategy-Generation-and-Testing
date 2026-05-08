#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high in 12h uptrend with volume spike.
# Short when price breaks below Donchian(20) low in 12h downtrend with volume spike.
# Uses 12h EMA(50) for trend filter and 40-period volume spike for confirmation.
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
# Works in both bull and bear by following 12h trend direction.

name = "4h_Donchian_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA(50) for trend filter
    close_12h_series = pd.Series(close_12h)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up = ema_50_12h[1:] > ema_50_12h[:-1]  # Rising EMA = uptrend
    trend_up = np.concatenate([[False], trend_up])  # Align with 12h index
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 40-period volume spike (1.8x EMA)
    vol_ema = pd.Series(volume).ewm(span=40, adjust=False, min_periods=40).mean().values
    vol_confirm = volume > (vol_ema * 1.8)
    
    # Align 12h indicators to 4h timeframe
    trend_up_aligned = align_htf_to_ltf(prices, df_12h, trend_up.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for Donchian and volume EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or
            np.isnan(trend_up_aligned[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Donchian breakout in 12h uptrend with volume
            if (trend_up_aligned[i] > 0.5 and  # 12h uptrend
                close[i] >= high_max[i] and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Donchian breakdown in 12h downtrend with volume
            elif (trend_up_aligned[i] <= 0.5 and  # 12h downtrend
                  close[i] <= low_min[i] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: reverse signal or stop at Donchian low
            if (trend_up_aligned[i] <= 0.5 or  # 12h downtrend
                close[i] <= low_min[i]):  # Break below Donchian low
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: reverse signal or stop at Donchian high
            if (trend_up_aligned[i] > 0.5 or  # 12h uptrend
                close[i] >= high_max[i]):  # Break above Donchian high
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals