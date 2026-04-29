#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ADX25 trend filter and volume confirmation
# Long when price breaks above Donchian upper(20) AND 1d ADX > 25 AND volume > 1.8x 20-bar avg
# Short when price breaks below Donchian lower(20) AND 1d ADX > 25 AND volume > 1.8x 20-bar avg
# Exit when price retests Donchian midpoint OR 1d ADX < 20 (trend weakening)
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 12-37 trades/year on 12h timeframe.
# Combines price structure breakout with daily trend strength filter to capture strong moves
# while avoiding false signals in ranging markets. Works in both bull (trend continuation) and bear (strong downtrends).

name = "12h_Donchian20_1dADX25_VolumeSpike_v1"
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
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14) for trend filter
    # ADX requires +DI, -DI, and TR calculations
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # first TR is NaN
    
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    tr_smoothed = wilders_smoothing(tr, period)
    plus_dm_smoothed = wilders_smoothing(plus_dm, period)
    minus_dm_smoothed = wilders_smoothing(minus_dm, period)
    
    # Avoid division by zero
    plus_di = np.where(tr_smoothed != 0, (plus_dm_smoothed / tr_smoothed) * 100, 0)
    minus_di = np.where(tr_smoothed != 0, (minus_dm_smoothed / tr_smoothed) * 100, 0)
    
    dx = np.where((plus_di + minus_di) != 0, 
                  np.abs((plus_di - minus_di) / (plus_di + minus_di)) * 100, 0)
    adx = wilders_smoothing(dx, period)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Donchian channels (20-period) on 12h timeframe
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # Donchian and ADX warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_adx = adx_aligned[i]
        curr_upper = donchian_upper[i]
        curr_lower = donchian_lower[i]
        curr_mid = donchian_mid[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price retests Donchian midpoint OR ADX < 20 (trend weakening)
            if curr_close <= curr_mid or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retests Donchian midpoint OR ADX < 20 (trend weakening)
            if curr_close >= curr_mid or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian upper AND ADX > 25 AND volume confirmation
            if curr_close > curr_upper and curr_adx > 25 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian lower AND ADX > 25 AND volume confirmation
            elif curr_close < curr_lower and curr_adx > 25 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals