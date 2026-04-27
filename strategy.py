#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with volume confirmation and 1d EMA(50) trend filter.
# Enters long when price breaks above 4h Donchian upper band with volume, short when breaks below lower band with volume.
# Uses 1d EMA(50) to filter trades in direction of daily trend. Designed for ~20-40 trades/year.
# Works in bull/bear: breaks above upper band indicate strength, breaks below lower band indicate weakness.
# Volume filter (volume > 1.5x 20-period average) avoids false breakouts.
# Exit when price returns to 4h Donchian middle band or daily trend changes.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channels (20-period)
    upper_4h = np.full(len(df_4h), np.nan)
    lower_4h = np.full(len(df_4h), np.nan)
    middle_4h = np.full(len(df_4h), np.nan)
    
    for i in range(19, len(df_4h)):
        upper_4h[i] = np.max(high_4h[i-19:i+1])
        lower_4h[i] = np.min(low_4h[i-19:i+1])
        middle_4h[i] = (upper_4h[i] + lower_4h[i]) / 2.0
    
    # Align 4h Donchian channels to 1h
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    middle_4h_aligned = align_htf_to_ltf(prices, df_4h, middle_4h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: volume > 1.5 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # 20% position size
    
    # Warmup: need Donchian (20), EMA (50), volume MA (20)
    start_idx = max(19, 50, 19)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or np.isnan(middle_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filters
        daily_bullish = price > ema_50_1d_aligned[i]
        daily_bearish = price < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above 4h upper band with volume and daily bullish
            if price > upper_4h_aligned[i] and vol_filter and daily_bullish:
                signals[i] = size
                position = 1
            # Short: price breaks below 4h lower band with volume and daily bearish
            elif price < lower_4h_aligned[i] and vol_filter and daily_bearish:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns below 4h middle band or daily trend turns bearish
            if price < middle_4h_aligned[i] or not daily_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns above 4h middle band or daily trend turns bullish
            if price > middle_4h_aligned[i] or not daily_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Donchian_Breakout_Volume_Trend"
timeframe = "1h"
leverage = 1.0