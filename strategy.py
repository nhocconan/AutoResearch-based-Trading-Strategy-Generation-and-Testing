#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and 1d ADX trend filter
# Long when price breaks above 20-period high AND volume > 1.8x 20-period average AND 1d ADX > 25 (trending market)
# Short when price breaks below 20-period low AND volume > 1.8x 20-period average AND 1d ADX > 25 (trending market)
# Exit when price crosses back to the opposite Donchian level (long exit at 20-period low, short exit at 20-period high)
# Uses discrete sizing (0.25) to limit fee drag. Target: 12-30 trades/year per symbol.
# Donchian channels provide clear breakout levels, volume confirms institutional participation,
# ADX filter ensures we only trade in trending markets to avoid whipsaws in ranging conditions.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "12h_Donchian20_VolumeSpike_1dADX_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Donchian channels, volume MA, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels on 1d data (using previous day's data to avoid look-ahead)
    prev_high = np.concatenate([[np.nan], df_1d['high'].values[:-1]])
    prev_low = np.concatenate([[np.nan], df_1d['low'].values[:-1]])
    
    # 20-period Donchian high and low
    donchian_high = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate volume MA on 1d data (using previous day's volume)
    prev_volume = np.concatenate([[np.nan], df_1d['volume'].values[:-1]])
    vol_ma_20 = pd.Series(prev_volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate ADX on 1d data
    # ADX calculation: +DM, -DM, TR, then smoothed
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.concatenate([[np.nan], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[np.nan], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = np.where((high_1d - np.concatenate([[np.nan], high_1d[:-1]])) > 
                       (np.concatenate([[np.nan], low_1d[:-1]]) - low_1d),
                       np.maximum(high_1d - np.concatenate([[np.nan], high_1d[:-1]]), 0), 0)
    minus_dm = np.where((np.concatenate([[np.nan], low_1d[:-1]]) - low_1d) > 
                        (high_1d - np.concatenate([[np.nan], high_1d[:-1]])),
                        np.maximum(np.concatenate([[np.nan], low_1d[:-1]]) - low_1d, 0), 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(values[:period])
            # Subsequent values: prev * (1 - 1/period) + current * (1/period)
            for i in range(period, len(values)):
                if not np.isnan(result[i-1]) and not np.isnan(values[i]):
                    result[i] = result[i-1] * (1 - 1/period) + values[i] * (1/period)
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    # Align indicators to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 1.8x 20-period average (spike filter)
    volume_spike = volume > (1.8 * volume_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND volume spike AND ADX > 25
            if (close[i] > donchian_high_aligned[i] and 
                volume_spike[i] and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND volume spike AND ADX > 25
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_spike[i] and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back to Donchian low
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back to Donchian high
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals