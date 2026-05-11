# 6h Donchian Breakout + Weekly Pivot Bias + Volume Spike
# Long when price breaks above Donchian(20) high + weekly pivot bias bullish + volume spike
# Short when price breaks below Donchian(20) low + weekly pivot bias bearish + volume spike
# Exit when price returns to Donchian midpoint or volatility drops
# Weekly pivot bias from previous week's close relative to weekly pivot point
# Weekly pivot point calculated from prior week's OHLC

#!/usr/bin/env python3
name = "6h_Donchian_Breakout_WeeklyPivotBias_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) for breakout signals
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Weekly pivot bias (from 1w data)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot point from prior week's OHLC
    # Pivot = (High + Low + Close) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Weekly bias: 1 if weekly close > pivot (bullish), -1 if weekly close < pivot (bearish)
    weekly_bias = np.where(weekly_close > weekly_pivot, 1, -1)
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias)
    
    # Volume filter: volume > 2.0 x 20-period average (to avoid noise)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_bias_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + weekly bias bullish + volume spike
            if close[i] > donchian_high[i] and weekly_bias_aligned[i] > 0 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + weekly bias bearish + volume spike
            elif close[i] < donchian_low[i] and weekly_bias_aligned[i] < 0 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to Donchian midpoint or volatility drops
            if close[i] < donchian_mid[i] or volume[i] < vol_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to Donchian midpoint or volatility drops
            if close[i] > donchian_mid[i] or volume[i] < vol_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals