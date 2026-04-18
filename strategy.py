#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with daily ADX trend filter and volume confirmation.
# Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength.
# Daily ADX > 25 ensures we only trade in strong trending markets, avoiding chop.
# Volume confirmation adds conviction to trend continuation signals.
# Designed for low trade frequency (20-50/year) to minimize fee drag in 4h timeframe.
# Works in bull markets (long when aligned above teeth) and bear markets (short when aligned below teeth).
name = "4h_WilliamsAlligator_ADX_TrendFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ADX filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Williams Alligator (13,8,5) smoothed with SMMA
    # Jaw (13-period), Teeth (8-period), Lips (5-period)
    def smoothed_moving_average(arr, period):
        sma = np.full_like(arr, np.nan)
        if len(arr) >= period:
            sma[period-1] = np.mean(arr[:period])
            for i in range(period, len(arr)):
                sma[i] = (sma[i-1] * (period-1) + arr[i]) / period
        return sma
    
    jaw = smoothed_moving_average(close, 13)
    teeth = smoothed_moving_average(close, 8)
    lips = smoothed_moving_average(close, 5)
    
    # Daily ADX (14-period) for trend strength
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # True Range
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_d[1:] - high_d[:-1]) > (low_d[:-1] - low_d[1:]), 
                       np.maximum(high_d[1:] - high_d[:-1], 0), 0)
    dm_minus = np.where((low_d[:-1] - low_d[1:]) > (high_d[1:] - high_d[:-1]), 
                        np.maximum(low_d[:-1] - low_d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+ and DM-
    def wilders_smoothing(arr, period):
        smoothed = np.full_like(arr, np.nan)
        if len(arr) >= period:
            smoothed[period-1] = np.nansum(arr[:period])
            for i in range(period, len(arr)):
                smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + arr[i]
        return smoothed
    
    tr_smoothed = wilders_smoothing(tr, 14)
    dm_plus_smoothed = wilders_smoothing(dm_plus, 14)
    dm_minus_smoothed = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr_smoothed != 0, 100 * dm_plus_smoothed / tr_smoothed, 0)
    di_minus = np.where(tr_smoothed != 0, 100 * dm_minus_smoothed / tr_smoothed, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # ADX threshold for trend strength
    adx_threshold = 25
    strong_trend = adx >= adx_threshold
    
    # Align daily ADX strong_trend to 4h timeframe
    strong_trend_aligned = align_htf_to_ltf(prices, df_1d, strong_trend.astype(float))
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(strong_trend_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        # Trend filter: daily ADX shows strong trend
        trend_filter = strong_trend_aligned[i] > 0
        
        if position == 0:
            # Long: Lips above Teeth above Jaw (bullish alignment) AND volume confirmation AND trend filter
            long_signal = lips[i] > teeth[i] and teeth[i] > jaw[i]
            if vol_confirm and trend_filter and long_signal:
                signals[i] = 0.25
                position = 1
            # Short: Lips below Teeth below Jaw (bearish alignment) AND volume confirmation AND trend filter
            elif vol_confirm and trend_filter and lips[i] < teeth[i] and teeth[i] < jaw[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Lips cross below Teeth OR trend weakens
            exit_condition = lips[i] < teeth[i] or strong_trend_aligned[i] <= 0
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Lips cross above Teeth OR trend weakens
            exit_condition = lips[i] > teeth[i] or strong_trend_aligned[i] <= 0
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals