#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume filter and 1w trend filter
# Donchian breakout captures trend continuation with clear entry/exit levels
# Volume filter ensures breakouts have institutional participation
# Weekly trend filter (price > 200 EMA) avoids counter-trend trades in strong trends
# Designed for low trade frequency (target 15-25/year) with high win rate
# Works in bull markets (breakouts with volume) and bear markets (avoids false breakouts via trend filter)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data once
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Load 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 12h Donchian channels (20-period)
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # 1d volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume_1d > (1.5 * vol_ma)
    
    # 1w trend filter: price > 200 EMA (bullish trend)
    ema200 = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    trend_filter = close_1w > ema200
    
    # Align indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    vol_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_filter)
    trend_filter_aligned = align_htf_to_ltf(prices, df_1w, trend_filter)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_filter_aligned[i]) or np.isnan(trend_filter_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian high with volume and trend confirmation
        if (close[i] > donchian_high_aligned[i] and 
            vol_filter_aligned[i] and 
            trend_filter_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = position_size
        
        # Short entry: price breaks below Donchian low with volume and counter-trend (for bear markets)
        elif (close[i] < donchian_low_aligned[i] and 
              vol_filter_aligned[i] and 
              not trend_filter_aligned[i] and  # In bear trend, trend filter is false
              position >= 0):
            position = -1
            signals[i] = -position_size
        
        # Exit: reverse signal or time-based exit (optional)
        elif position == 1 and close[i] < donchian_low_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > donchian_high_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian20_Volume_TrendFilter"
timeframe = "12h"
leverage = 1.0