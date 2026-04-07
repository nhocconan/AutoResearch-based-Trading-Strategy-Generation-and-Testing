#!/usr/bin/env python3
"""
4h_camarilla_pivot_1d_volume_v1
Hypothesis: On 4h timeframe, use Camarilla pivot levels from 1d timeframe to identify key support/resistance zones. Enter long when price bounces above S3 with volume confirmation, and enter short when price rejects from R3 with volume confirmation. Filter trades using 1w ADX to avoid low-momentum environments. This strategy aims to capture mean-reversion bounces at strong institutional levels while avoiding choppy markets. Target: 20-50 trades per year to minimize fee drag and work in both bull and bear markets by fading extremes at proven pivot levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_1d_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_S3 = np.zeros(len(df_1d))
    camarilla_S4 = np.zeros(len(df_1d))
    camarilla_R3 = np.zeros(len(df_1d))
    camarilla_R4 = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        if i == 0:
            camarilla_S3[i] = camarilla_S4[i] = camarilla_R3[i] = camarilla_R4[i] = np.nan
        else:
            # Use previous day's OHLC
            high_prev = high_1d[i-1]
            low_prev = low_1d[i-1]
            close_prev = close_1d[i-1]
            
            range_prev = high_prev - low_prev
            camarilla_S3[i] = close_prev - 1.1 * range_prev / 6
            camarilla_S4[i] = close_prev - 1.1 * range_prev / 2
            camarilla_R3[i] = close_prev + 1.1 * range_prev / 6
            camarilla_R4[i] = close_prev + 1.1 * range_prev / 2
    
    # Align Camarilla levels to 4h
    S3_4h = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    S4_4h = align_htf_to_ltf(prices, df_1d, camarilla_S4)
    R3_4h = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    R4_4h = align_htf_to_ltf(prices, df_1d, camarilla_R4)
    
    # Calculate 1w ADX for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.concatenate([[np.max([high_1w[0] - low_1w[0], np.abs(high_1w[0] - close_1w[0]), np.abs(low_1w[0] - close_1w[0])])],
                            np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]),
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]),
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
    def wilder_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
        for i in range(period, len(data)):
            result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
        return result
    
    atr_1w = wilder_smoothing(tr_1w, 14)
    dm_plus_smooth = wilder_smoothing(dm_plus, 14)
    dm_minus_smooth = wilder_smoothing(dm_minus, 14)
    
    # Calculate DI+ and DI-
    di_plus = np.where(atr_1w > 0, dm_plus_smooth / atr_1w * 100, 0)
    di_minus = np.where(atr_1w > 0, dm_minus_smooth / atr_1w * 100, 0)
    
    # Calculate DX and ADX
    dx = np.where((di_plus + di_minus) > 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx_1w = wilder_smoothing(dx, 14)
    
    # Align ADX to 4h
    adx_1w_4h = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate volume moving average (20-period)
    def sma_with_min_period(data, period):
        result = np.full(len(data), np.nan)
        for i in range(len(data)):
            if i >= period - 1:
                result[i] = np.mean(data[i-period+1:i+1])
        return result
    
    vol_ma = sma_with_min_period(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any data is not available
        if (np.isnan(S3_4h[i]) or np.isnan(R3_4h[i]) or np.isnan(adx_1w_4h[i]) or
            np.isnan(vol_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        # ADX filter: only trade when ADX > 25 (trending market)
        adx_ok = adx_1w_4h[i] > 25
        
        if position == 1:  # Long position
            # Exit: price closes below S4 (strong breakdown)
            if close[i] < S4_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R4 (strong breakout)
            if close[i] > R4_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation and ADX filter
            if volume_ok and adx_ok:
                # Long entry: price crosses above S3 with volume
                if close[i] > S3_4h[i] and close[i-1] <= S3_4h[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price crosses below R3 with volume
                elif close[i] < R3_4h[i] and close[i-1] >= R3_4h[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals