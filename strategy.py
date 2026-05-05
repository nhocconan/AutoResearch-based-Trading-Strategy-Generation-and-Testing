#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d ADX trend filter
# Long when: price breaks above Donchian upper channel AND volume > 1.5x 20-period MA AND 1d ADX > 25
# Short when: price breaks below Donchian lower channel AND volume > 1.5x 20-period MA AND 1d ADX > 25
# Exit when: price touches Donchian middle line OR ADX < 20 (trend weakens)
# Uses Donchian for structure, volume for conviction, ADX for regime filter
# Timeframe: 4h, HTF: 1d for ADX. Target: 100-200 total trades over 4 years (25-50/year) to avoid fee drag.

name = "4h_Donchian20_1dADX_VolumeConfirm"
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
    
    # Calculate Donchian channels on 4h
    lookback = 20
    if len(high) >= lookback:
        upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
        lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
        middle = (upper + lower) / 2.0
    else:
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        middle = np.full(n, np.nan)
    
    # Volume confirmation on 4h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # need sufficient data for ADX
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    if len(high_1d) >= 14:
        # True Range
        tr1 = np.abs(high_1d[1:] - low_1d[1:])
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[np.nan], tr])  # align with index
        
        # Directional Movement
        up_move = high_1d[1:] - high_1d[:-1]
        down_move = low_1d[:-1] - low_1d[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
        def wilder_smooth(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1]/period) + data[i]
            return result
        
        atr = wilder_smooth(tr, 14)
        plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
        minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilder_smooth(dx, 14)
    else:
        adx = np.full(len(df_1d), np.nan)
    
    # ADX trend filter: ADX > 25 = strong trend
    adx_trend = adx > 25
    adx_weak = adx < 20  # for exit condition
    
    # Align 1d ADX to 4h timeframe
    adx_trend_aligned = align_htf_to_ltf(prices, df_1d, adx_trend.astype(float))
    adx_weak_aligned = align_htf_to_ltf(prices, df_1d, adx_weak.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(adx_trend_aligned[i]) or 
            np.isnan(adx_weak_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above upper + volume filter + strong trend
            if (close[i] > upper[i] and 
                volume_filter[i] and 
                adx_trend_aligned[i] == 1.0):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below lower + volume filter + strong trend
            elif (close[i] < lower[i] and 
                  volume_filter[i] and 
                  adx_trend_aligned[i] == 1.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches middle OR trend weakens
            if (close[i] >= middle[i] or adx_weak_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches middle OR trend weakens
            if (close[i] <= middle[i] or adx_weak_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals