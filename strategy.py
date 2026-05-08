#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6s Donchian breakout with 1d trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high + 1d EMA50 up + volume > 1.5x average.
# Short when price breaks below Donchian(20) low + 1d EMA50 down + volume > 1.5x average.
# Exit when price crosses Donchian midline or trend reverses.
# Works in bull (breakouts with trend) and bear (breakouts against trend filtered by 1d EMA).
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost.

name = "6s_Donchian_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # 1d EMA(50) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_slope = ema_50_1d[1:] > ema_50_1d[:-1]
    ema_50_1d_slope = np.concatenate([[False], ema_50_1d_slope])
    
    # Align 1d EMA slope to 6h
    ema_50_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d_slope.astype(float))
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_50_1d_slope_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > Donchian high + 1d EMA up + volume filter
            if (close[i] > highest_high[i] and 
                ema_50_1d_slope_aligned[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: price < Donchian low + 1d EMA down + volume filter
            elif (close[i] < lowest_low[i] and 
                  not ema_50_1d_slope_aligned[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses Donchian midline OR trend reverses
            if (close[i] < donchian_mid[i] or not ema_50_1d_slope_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses Donchian midline OR trend reverses
            if (close[i] > donchian_mid[i] or ema_50_1d_slope_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals