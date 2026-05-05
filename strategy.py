#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1d ADX(14) trend filter
# Long when: price breaks above 20-period high, volume > 1.5x 20-period average, and 1d ADX > 25
# Short when: price breaks below 20-period low, volume > 1.5x 20-period average, and 1d ADX > 25
# Exit when price returns to midpoint of Donchian channel (mean reversion)
# Uses Donchian structure for breakouts in trending markets and mean reversion in ranges.
# Timeframe: 4h, HTF: 1d. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Donchian20_Breakout_1dADX25_VolumeSpike"
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
    
    # Calculate volume confirmation on 4h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for ADX trend filter and Donchian levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14) trend filter
    if len(high_1d) >= 14:
        # True Range
        tr1 = np.abs(high_1d[1:] - low_1d[1:])
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with index
        
        # Directional Movement
        up_move = np.diff(high_1d)
        down_move = -np.diff(low_1d)
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[np.nan], plus_dm])
        minus_dm = np.concatenate([[np.nan], minus_dm])
        
        # Smoothed values using Wilder's smoothing (equivalent to EMA with alpha=1/period)
        def wilders_smoothing(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
            return result
        
        atr = wilders_smoothing(tr, 14)
        plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
        minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilders_smoothing(dx, 14)
        
        adx_filter = adx > 25
    else:
        adx_filter = np.zeros(len(close_1d), dtype=bool)
    
    # Align ADX filter to 4h timeframe
    adx_filter_aligned = align_htf_to_ltf(prices, df_1d, adx_filter.astype(float)) > 0.5
    
    # Calculate Donchian(20) levels from previous 1d bar
    if len(high_1d) >= 20:
        # Rolling max/min of high/low over 20 periods
        def rolling_max(arr, window):
            result = np.full_like(arr, np.nan)
            for i in range(window-1, len(arr)):
                result[i] = np.nanmax(arr[i-window+1:i+1])
            return result
        
        def rolling_min(arr, window):
            result = np.full_like(arr, np.nan)
            for i in range(window-1, len(arr)):
                result[i] = np.nanmin(arr[i-window+1:i+1])
            return result
        
        donchian_high = rolling_max(high_1d, 20)
        donchian_low = rolling_min(low_1d, 20)
        donchian_mid = (donchian_high + donchian_low) / 2
    else:
        donchian_high = np.full(len(close_1d), np.nan)
        donchian_low = np.full(len(close_1d), np.nan)
        donchian_mid = np.full(len(close_1d), np.nan)
    
    # Align Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_filter_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, volume filter, and ADX > 25
            if (close[i] > donchian_high_aligned[i] and 
                volume_filter[i] and 
                adx_filter_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low, volume filter, and ADX > 25
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_filter[i] and 
                  adx_filter_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below Donchian midpoint (mean reversion)
            if close[i] < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian midpoint (mean reversion)
            if close[i] > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals