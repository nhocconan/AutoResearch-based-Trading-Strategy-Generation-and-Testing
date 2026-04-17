#!/usr/bin/env python3
"""
4h_ADX_Trend_Plus_VolumeBreakout_V1
Hypothesis: Use ADX to detect strong trending conditions (ADX > 25) on 4h timeframe.
Enter long when price breaks above 4h high of previous 10 bars with volume confirmation (>1.5x average volume).
Enter short when price breaks below 4h low of previous 10 bars with volume confirmation.
Exit when ADX falls below 20 (trend weakening) or opposite breakout occurs.
Uses volume breakout to capture momentum bursts in trending markets, works in both bull and bear.
Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === 4h Data (HTF for trend confirmation) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate ADX on 4h (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align length
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[np.nan], plus_dm])
        minus_dm = np.concatenate([[np.nan], minus_dm])
        
        # Smoothed values
        def smooth_wilder(arr, period):
            result = np.full_like(arr, np.nan)
            if len(arr) < period:
                return result
            # First value is simple average
            result[period-1] = np.nansum(arr[1:period])  # skip first nan
            # Wilder smoothing
            for i in range(period, len(arr)):
                if not np.isnan(arr[i]) and not np.isnan(result[i-1]):
                    result[i] = (result[i-1] * (period-1) + arr[i]) / period
            return result
        
        atr = smooth_wilder(tr, period)
        plus_di = 100 * smooth_wilder(plus_dm, period) / atr
        minus_di = 100 * smooth_wilder(minus_dm, period) / atr
        dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = smooth_wilder(dx, period)
        return adx
    
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, 14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Calculate 4h rolling high/low for breakout (10-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i < window - 1:
                continue
            window_slice = arr[max(0, i-window+1):i+1]
            if np.all(np.isnan(window_slice)):
                result[i] = np.nan
            else:
                result[i] = np.nanmax(window_slice)
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i < window - 1:
                continue
            window_slice = arr[max(0, i-window+1):i+1]
            if np.all(np.isnan(window_slice)):
                result[i] = np.nan
            else:
                result[i] = np.nanmin(window_slice)
        return result
    
    high_10_4h = rolling_max(high_4h, 10)
    low_10_4h = rolling_min(low_4h, 10)
    high_10_4h_aligned = align_htf_to_ltf(prices, df_4h, high_10_4h)
    low_10_4h_aligned = align_htf_to_ltf(prices, df_4h, low_10_4h)
    
    # Volume confirmation on 4h
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(adx_4h_aligned[i]) or 
            np.isnan(high_10_4h_aligned[i]) or
            np.isnan(low_10_4h_aligned[i]) or
            np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 4h bar's volume for confirmation
        vol_4h_current = align_htf_to_ltf(prices, df_4h, volume_4h)[i]
        vol_confirmed = vol_4h_current > 1.5 * vol_ma_4h_aligned[i]
        
        # Trend filter: only trade when ADX > 25 (strong trend)
        strong_trend = adx_4h_aligned[i] > 25
        
        # Exit when trend weakens (ADX < 20)
        weak_trend = adx_4h_aligned[i] < 20
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above 10-period high with volume confirmation and strong trend
            if close[i] > high_10_4h_aligned[i] and vol_confirmed and strong_trend:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below 10-period low with volume confirmation and strong trend
            elif close[i] < low_10_4h_aligned[i] and vol_confirmed and strong_trend:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit conditions: trend weakening OR opposite breakout
            if weak_trend or close[i] < low_10_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: trend weakening OR opposite breakout
            if weak_trend or close[i] > high_10_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_ADX_Trend_Plus_VolumeBreakout_V1"
timeframe = "4h"
leverage = 1.0