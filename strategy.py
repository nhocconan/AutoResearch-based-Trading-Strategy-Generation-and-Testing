#!/usr/bin/env python3
"""
12h_Linear_Regression_Channel_Breakout
Hypothesis: Linear regression channel identifies mean-reversion extremes in 12h timeframe. 
Long when price touches lower channel band with volume confirmation in ranging markets.
Short when price touches upper channel band with volume confirmation in ranging markets.
Uses 1d ADX < 25 to filter for ranging conditions and avoid trending whipsaws.
Target: 15-30 trades/year per symbol.
"""

name = "12h_Linear_Regression_Channel_Breakout"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Linear regression channel on 12h data (50-period)
    def linear_regression_channel(arr, period):
        """Returns upper, lower bands of linear regression channel"""
        if len(arr) < period:
            return np.full(len(arr), np.nan), np.full(len(arr), np.nan)
        
        upper = np.full(len(arr), np.nan)
        lower = np.full(len(arr), np.nan)
        
        for i in range(period-1, len(arr)):
            y = arr[i-period+1:i+1]
            x = np.arange(len(y))
            if len(y) < 2:
                continue
            # Calculate linear regression: y = mx + b
            A = np.vstack([x, np.ones(len(x))]).T
            m, b = np.linalg.lstsq(A, y, rcond=None)[0]
            # Current value
            current_y = m * (len(x)-1) + b
            # Standard deviation of residuals
            y_pred = m * x + b
            residuals = y - y_pred
            std_res = np.std(residuals) if len(residuals) > 1 else 0
            # Channel width: 1.5 * std deviation
            channel_width = 1.5 * std_res
            upper[i] = current_y + channel_width
            lower[i] = current_y - channel_width
        
        return upper, lower
    
    # Calculate 12h linear regression channel
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    upper_12h, lower_12h = linear_regression_channel(close_12h, 50)
    
    # Align to 12h timeframe
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    
    # 1d ADX for ranging filter (< 25 = ranging)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = np.concatenate([[np.nan], high_1d[1:] - high_1d[:-1]])
    down_move = np.concatenate([[np.nan], low_1d[:-1] - low_1d[1:]])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR and DM
    def smooth_wilder(arr, period):
        """Wilder's smoothing (equivalent to EMA with alpha=1/period)"""
        if len(arr) < period:
            return np.full(len(arr), np.nan)
        result = np.full(len(arr), np.nan)
        # First value is simple average
        result[period-1] = np.nanmean(arr[1:period])
        # Subsequent values: smoothed = prev * (1-1/period) + current * (1/period)
        for i in range(period, len(arr)):
            if np.isnan(result[i-1]) or np.isnan(arr[i]):
                result[i] = np.nan
            else:
                result[i] = result[i-1] * (1 - 1/period) + arr[i] * (1/period)
        return result
    
    period = 14
    tr_smooth = smooth_wilder(tr, period)
    plus_dm_smooth = smooth_wilder(plus_dm, period)
    minus_dm_smooth = smooth_wilder(minus_dm, period)
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = smooth_wilder(dx, period)
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_ma[:10] = vol_ma[-10:] = np.nan  # Handle edges
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_12h_aligned[i]) or np.isnan(lower_12h_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Ranging market condition: ADX < 25
        ranging = adx_aligned[i] < 25
        
        if position == 0:
            # LONG: price touches lower band + volume spike + ranging market
            if close[i] <= lower_12h_aligned[i] and volume_spike[i] and ranging:
                signals[i] = 0.25
                position = 1
            # SHORT: price touches upper band + volume spike + ranging market
            elif close[i] >= upper_12h_aligned[i] and volume_spike[i] and ranging:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price reaches middle (regression line) or opposite band
            # Calculate middle line (linear regression value)
            if not np.isnan(upper_12h_aligned[i]) and not np.isnan(lower_12h_aligned[i]):
                middle = (upper_12h_aligned[i] + lower_12h_aligned[i]) / 2
                if close[i] >= middle or close[i] >= upper_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # EXIT SHORT: price reaches middle or opposite band
            if not np.isnan(upper_12h_aligned[i]) and not np.isnan(lower_12h_aligned[i]):
                middle = (upper_12h_aligned[i] + lower_12h_aligned[i]) / 2
                if close[i] <= middle or close[i] <= lower_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = 0.0
                position = 0
    
    return signals