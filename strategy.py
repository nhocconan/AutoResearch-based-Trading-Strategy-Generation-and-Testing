#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d ADX(14) trend filter
# Long when price breaks above Donchian upper AND volume > 1.5x 20-period average AND 1d ADX > 25 (trending market)
# Short when price breaks below Donchian lower AND volume > 1.5x 20-period average AND 1d ADX > 25 (trending market)
# Exit when price crosses back to Donchian midpoint OR ADX < 20 (trend weakening)
# Uses discrete sizing (0.25) to limit fee drag. Target: 20-40 trades/year per symbol.
# Donchian channels provide clear breakout levels, volume confirms conviction,
# ADX filters for trending conditions to avoid whipsaws in ranging markets.
# Works in bull markets via breakout longs and bear markets via breakdown shorts.

name = "4h_Donchian20_VolumeSpike_1dADX25_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])) > 
                       (np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d),
                       np.maximum(high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]]), 0), 0)
    dm_minus = np.where((np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d) > 
                        (high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])),
                        np.maximum(np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d, 0), 0)
    
    # Smoothed values
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) >= period:
            result[period-1] = np.nansum(arr[:period])
            for i in range(period, len(arr)):
                result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr = smma(tr, 14)
    dm_plus_smooth = smma(dm_plus, 14)
    dm_minus_smooth = smma(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, (dm_plus_smooth / atr) * 100, 0)
    di_minus = np.where(atr > 0, (dm_minus_smooth / atr) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = smma(dx, 14)
    
    # Trend filter: ADX > 25
    trending = adx > 25
    weak_trend = adx < 20  # Exit condition
    
    # Align 1d ADX to 4h timeframe
    trending_aligned = align_htf_to_ltf(prices, df_1d, trending.astype(float))
    weak_trend_aligned = align_htf_to_ltf(prices, df_1d, weak_trend.astype(float))
    
    # Donchian(20) on 4h data
    lookback = 20
    if len(high) >= lookback:
        upper = np.full_like(high, np.nan, dtype=float)
        lower = np.full_like(high, np.nan, dtype=float)
        for i in range(lookback-1, len(high)):
            upper[i] = np.max(high[i-lookback+1:i+1])
            lower[i] = np.min(low[i-lookback+1:i+1])
        midpoint = (upper + lower) / 2
    else:
        upper = lower = midpoint = np.full_like(high, np.nan, dtype=float)
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(upper[i]) or 
            np.isnan(lower[i]) or 
            np.isnan(trending_aligned[i]) or 
            np.isnan(weak_trend_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper AND volume spike AND trending market
            if (close[i] > upper[i] and 
                volume_filter[i] and 
                trending_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower AND volume spike AND trending market
            elif (close[i] < lower[i] and 
                  volume_filter[i] and 
                  trending_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back to midpoint OR trend weakens
            if (close[i] < midpoint[i] or 
                weak_trend_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back to midpoint OR trend weakens
            if (close[i] > midpoint[i] or 
                weak_trend_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals