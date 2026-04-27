#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator (Jaw/Teeth/Lips) crossover with 1d ADX trend filter and volume confirmation
# Uses Bill Williams' Alligator indicator to detect trend emergence (lips crossing teeth/jaw).
# Only takes signals in direction of 1d ADX > 25 (trending regime).
# Volume > 1.5x 20-period average confirms momentum.
# Works in bull/bear markets by following higher timeframe trend.
# Target: 20-30 trades/year to minimize fee decay while capturing sustained trends.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate ADX on 1d (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    n_1d = len(close_1d)
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing function
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Rest is Wilder's smoothing
        for i in range(period, len(data)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period_adx = 14
    tr_smooth = wilders_smooth(tr, period_adx)
    dm_plus_smooth = wilders_smooth(dm_plus, period_adx)
    dm_minus_smooth = wilders_smooth(dm_minus, period_adx)
    
    # DI+ and DI-
    di_plus = np.full(n_1d, np.nan)
    di_minus = np.full(n_1d, np.nan)
    dx = np.full(n_1d, np.nan)
    
    for i in range(len(tr_smooth)):
        if np.isnan(tr_smooth[i]) or tr_smooth[i] == 0:
            di_plus[i] = 0
            di_minus[i] = 0
        else:
            di_plus[i] = 100 * dm_plus_smooth[i] / tr_smooth[i]
            di_minus[i] = 100 * dm_minus_smooth[i] / tr_smooth[i]
    
    for i in range(len(dx)):
        if np.isnan(di_plus[i]) or np.isnan(di_minus[i]) or (di_plus[i] + di_minus[i]) == 0:
            dx[i] = 0
        else:
            dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX = Wilder's smoothed DX
    adx_1d = wilders_smooth(dx, period_adx)
    
    # Williams Alligator on 4h (13,8,5 smoothed medians)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    # Calculate smoothed medians (using Wilder's smoothing on median price)
    median_price = (high + low) / 2
    
    def median_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Rest is Wilder's smoothing
        for i in range(period, len(data)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    jaw = median_smooth(median_price, jaw_period)
    teeth = median_smooth(median_price, teeth_period)
    lips = median_smooth(median_price, lips_period)
    
    # 20-period average volume for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # Align HTF indicators to 4h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(jaw_period, teeth_period, lips_period, vol_period) + period_adx
    
    for i in range(start_idx, n):
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Conditions:
        # 1. Alligator alignment: lips > teeth > jaw (bullish) or lips < teeth < jaw (bearish)
        # 2. ADX > 25: trending regime on 1d
        # 3. Volume confirmation: > 1.5x average volume
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        trend_condition = adx_1d_aligned[i] > 25
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long: bullish alignment with trend and volume
            if bullish_alignment and trend_condition and volume_confirmation:
                signals[i] = size
                position = 1
            # Short: bearish alignment with trend and volume
            elif bearish_alignment and trend_condition and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: alignment breaks (lips < teeth) or trend weakens
            if lips[i] < teeth[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: alignment breaks (lips > teeth) or trend weakens
            if lips[i] > teeth[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_WilliamsAlligator_ADXTrend_Volume"
timeframe = "4h"
leverage = 1.0