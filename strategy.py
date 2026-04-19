#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1-day Trend Index (ADX14) and Donchian breakout (14-period) with volume confirmation.
# Only trades when ADX > 25 (trending market) and volume > 1.5x 20-period average.
# Uses discrete position sizing (0.25) to limit trades and avoid overtrading.
# Target: 20-40 trades/year to stay under fee drag threshold.
name = "4h_1d_ADX14_Donchian14_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for ADX14 trend (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[1:period]) if not np.all(np.isnan(arr[1:period])) else np.nan
        # Wilder smoothing
        for i in range(period, len(arr)):
            if np.isnan(result[i-1]) or np.isnan(arr[i]):
                result[i] = np.nan
            else:
                result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr = smooth_wilder(tr, 14)
    plus_di = 100 * smooth_wilder(plus_dm, 14) / atr
    minus_di = 100 * smooth_wilder(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth_wilder(dx, 14)
    adx_1d = adx
    
    # Align ADX to 4h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Get 4h data for Donchian14 breakout (called ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    # Donchian channels: 14-period high/low
    high_14_4h = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    low_14_4h = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    high_14_4h_aligned = align_htf_to_ltf(prices, df_4h, high_14_4h)
    low_14_4h_aligned = align_htf_to_ltf(prices, df_4h, low_14_4h)
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(high_14_4h_aligned[i]) or 
            np.isnan(low_14_4h_aligned[i]) or np.isnan(volume_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: ADX > 25 (trending) AND price breaks 4h Donchian high with volume
            if (adx_1d_aligned[i] > 25 and 
                close[i] > high_14_4h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25 (trending) AND price breaks 4h Donchian low with volume
            elif (adx_1d_aligned[i] > 25 and 
                  close[i] < low_14_4h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if ADX < 20 (no trend) or price breaks below 4h Donchian low
            if adx_1d_aligned[i] < 20 or close[i] < low_14_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if ADX < 20 (no trend) or price breaks above 4h Donchian high
            if adx_1d_aligned[i] < 20 or close[i] > high_14_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals