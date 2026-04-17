#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot bias and volume confirmation.
Long when price breaks above Donchian upper band AND price is above 1d weekly pivot (PP) AND volume > 1.5x 20-period average.
Short when price breaks below Donchian lower band AND price is below 1d weekly pivot (PP) AND volume > 1.5x 20-period average.
Exit when price crosses the Donchian middle band (20-period mean).
Weekly pivot derived from 1d OHLC: PP = (weekly_high + weekly_low + weekly_close) / 3.
Designed for low trade frequency (12-37/year) on 6h timeframe with strong trend following edge and weekly structure bias.
Works in bull markets via breakouts and in bear markets via short breakdowns with pivot filter reducing false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Donchian calculation (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate Donchian channels (20-period)
    def rolling_max(arr, window):
        """Rolling maximum"""
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        """Rolling minimum"""
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    def rolling_mean(arr, window):
        """Rolling mean"""
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            result[i] = np.mean(arr[i-window+1:i+1])
        return result
    
    upper_band = rolling_max(high_6h, 20)
    lower_band = rolling_min(low_6h, 20)
    middle_band = rolling_mean(close_6h, 20)
    
    # Calculate volume average (20-period) on 6h
    volume_6h_series = pd.Series(volume_6h)
    volume_ma_6h = volume_6h_series.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for weekly pivot calculation (higher timeframe)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot point from 1d data
    # Weekly high = max of last 7 daily highs
    # Weekly low = min of last 7 daily lows
    # Weekly close = last daily close
    def rolling_max_weekly(arr):
        """Rolling maximum over 7 periods"""
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(6, len(arr)):
            result[i] = np.max(arr[i-6:i+1])
        return result
    
    def rolling_min_weekly(arr):
        """Rolling minimum over 7 periods"""
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(6, len(arr)):
            result[i] = np.min(arr[i-6:i+1])
        return result
    
    weekly_high = rolling_max_weekly(high_1d)
    weekly_low = rolling_min_weekly(low_1d)
    weekly_close = close_1d  # weekly close is the last daily close
    
    # Weekly pivot point: PP = (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align all indicators to 6h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_6h, upper_band)
    lower_aligned = align_htf_to_ltf(prices, df_6h, lower_band)
    middle_aligned = align_htf_to_ltf(prices, df_6h, middle_band)
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_6h)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(middle_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(weekly_pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        upper = upper_aligned[i]
        lower = lower_aligned[i]
        middle = middle_aligned[i]
        vol_ma = volume_ma_aligned[i]
        weekly_pivot = weekly_pivot_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: Breakout above upper band + price > weekly pivot + volume spike
            if price > upper and price > weekly_pivot and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below lower band + price < weekly pivot + volume spike
            elif price < lower and price < weekly_pivot and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit conditions for long
            if price < middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions for short
            if price > middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dWeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0