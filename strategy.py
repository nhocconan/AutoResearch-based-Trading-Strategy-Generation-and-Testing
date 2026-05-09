#!/usr/bin/env python3
# 4h_Donchian20_Volume_Trend_1dFilter
# Strategy: Donchian(20) breakout with volume confirmation and 1d EMA50 trend filter
# Long when price breaks above Donchian high(20) and volume > 1.5x avg volume and price > 1d EMA50
# Short when price breaks below Donchian low(20) and volume > 1.5x avg volume and price < 1d EMA50
# Exit when price crosses back below Donchian midpoint (mean reversion in ranges)
# Uses trend filter to avoid counter-trend trades and volume to confirm breakouts
# Designed for 4h timeframe with selective entries to minimize trade frequency

name = "4h_Donchian20_Volume_Trend_1dFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Donchian channels (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate average volume (20-period)
    def rolling_mean(arr, window):
        result = np.full_like(arr, np.nan)
        if len(arr) >= window:
            result[window-1] = np.mean(arr[:window])
            for i in range(window, len(arr)):
                result[i] = result[i-1] + (arr[i] - arr[i-window]) / window
        return result
    
    avg_volume = rolling_mean(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: breakout above Donchian high with volume confirmation and uptrend filter
            if (close[i] > donchian_high[i] and 
                volume[i] > 1.5 * avg_volume[i] and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: breakout below Donchian low with volume confirmation and downtrend filter
            elif (close[i] < donchian_low[i] and 
                  volume[i] > 1.5 * avg_volume[i] and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint (mean reversion in ranges)
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint (mean reversion in ranges)
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals