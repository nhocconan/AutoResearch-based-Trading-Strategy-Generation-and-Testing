#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Uses 4h Donchian channels for breakout signals in trending markets
# Requires price to break above/below 20-period Donchian channel
# Uses 1d EMA(50) to filter for trend direction (only long when price > EMA50, short when price < EMA50)
# Volume confirmation (>1.5x 20-bar average) ensures participation
# Designed for 4h timeframe to target 75-200 total trades over 4 years (19-50/year)
# Works in both bull/bear: captures breakouts in trending markets, avoids false signals in weak trends

name = "4h_Donchian20_1dEMA50_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h Donchian(20) channels
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume confirmation filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align HTF indicators to 4h timeframe (primary)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above Donchian upper band AND above 1d EMA50 AND volume confirmation
            if (close[i] > high_max_20[i] and close[i] > ema_50_1d_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower band AND below 1d EMA50 AND volume confirmation
            elif (close[i] < low_min_20[i] and close[i] < ema_50_1d_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian lower band
            if close[i] < low_min_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian upper band
            if close[i] > high_max_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals