#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12-hour Donchian breakout with volume confirmation and 1-day trend filter.
# Donchian breakouts capture momentum, while volume confirmation filters false breakouts.
# 1-day trend filter ensures alignment with higher timeframe momentum.
# Designed for low trade frequency (15-30/year) to minimize whipsaw and capture high-probability breakouts.
# Works in both bull and bear markets by using trend filter to only take breakouts in direction of higher timeframe trend.

name = "6h_DonchianBreakout_Volume_TrendFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian channels (20-period high/low)
    high_20 = np.full_like(high_12h, np.nan)
    low_20 = np.full_like(low_12h, np.nan)
    
    for i in range(19, len(high_12h)):
        high_20[i] = np.max(high_12h[i-19:i+1])
        low_20[i] = np.min(low_12h[i-19:i+1])
    
    # Align Donchian levels to 6h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_12h, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_12h, low_20)
    
    # Get 1d trend filter (EMA crossover)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up = ema_20 > ema_50  # Bullish when 20 EMA > 50 EMA
    trend_aligned = align_htf_to_ltf(prices, df_1d, trend_up.astype(float))
    
    # Volume confirmation: current volume > 2.0x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or
            np.isnan(trend_aligned[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above 20-period high with volume and uptrend
            if (close[i] > high_20_aligned[i] and
                trend_aligned[i] > 0.5 and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below 20-period low with volume and downtrend
            elif (close[i] < low_20_aligned[i] and
                  trend_aligned[i] <= 0.5 and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below 20-period low or trend turns down
            if close[i] < low_20_aligned[i] or trend_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above 20-period high or trend turns up
            if close[i] > high_20_aligned[i] or trend_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals