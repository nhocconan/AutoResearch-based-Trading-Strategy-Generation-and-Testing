#!/usr/bin/env python3
# 1h_4h1d_Trend_Slope_Filter
# Hypothesis: Use 4h and 1d linear regression slopes to determine trend direction, and enter long/short on 1h when price pulls back to 20-period EMA with volume confirmation.
# In bull markets, 4h/1d slopes are positive; in bear markets, negative. Pullbacks to EMA with volume offer high-probability entries.
# Trend slope filters avoid whipsaws; volume confirms institutional interest. Designed for low trade frequency (15-30/year) on 1h.
# Works in both bull and bear by following the higher timeframe trend.

name = "1h_4h1d_Trend_Slope_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def _linreg_slope(arr, window):
    """Calculate linear regression slope over window, returns array same length as input."""
    if len(arr) < window:
        return np.full_like(arr, np.nan, dtype=np.float64)
    slope = np.empty_like(arr, dtype=np.float64)
    slope[:] = np.nan
    for i in range(window - 1, len(arr)):
        y = arr[i - window + 1:i + 1]
        x = np.arange(window)
        if np.all(np.isnan(y)):
            slope[i] = np.nan
        else:
            # Use only non-nan points for regression
            mask = ~np.isnan(y)
            if np.sum(mask) < 2:
                slope[i] = np.nan
            else:
                x_valid = x[mask]
                y_valid = y[mask]
                slope[i] = (len(x_valid) * np.sum(x_valid * y_valid) - np.sum(x_valid) * np.sum(y_valid)) / (len(x_valid) * np.sum(x_valid ** 2) - np.sum(x_valid) ** 2)
    return slope

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 4h close linear regression slope (20-period)
    slope_4h = _linreg_slope(df_4h['close'].values, 20)
    slope_4h_aligned = align_ltf_to_htf(prices, df_4h, slope_4h)
    
    # Calculate 1d close linear regression slope (20-period)
    slope_1d = _linreg_slope(df_1d['close'].values, 20)
    slope_1d_aligned = align_ltf_to_htf(prices, df_1d, slope_1d)
    
    # 1h 20-period EMA for entry
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).values
    
    # Volume confirmation: 20-period volume average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 4h/1d slopes, EMA, volume MA
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(slope_4h_aligned[i]) or np.isnan(slope_1d_aligned[i]) or 
            np.isnan(ema_20[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: both 4h and 1d slopes must agree
        uptrend = slope_4h_aligned[i] > 0 and slope_1d_aligned[i] > 0
        downtrend = slope_4h_aligned[i] < 0 and slope_1d_aligned[i] < 0
        
        # Volume confirmation
        volume_confirm = volume[i] > vol_ma20[i] * 1.5
        
        # Entry conditions: price near 20 EMA (within 0.5%)
        near_ema = abs(close[i] - ema_20[i]) / ema_20[i] < 0.005
        
        if position == 0:
            # Long entry: uptrend on 4h/1d + price near EMA + volume
            if uptrend and near_ema and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short entry: downtrend on 4h/1d + price near EMA + volume
            elif downtrend and near_ema and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: trend breaks down or price moves far from EMA
            if not uptrend or abs(close[i] - ema_20[i]) / ema_20[i] > 0.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: trend breaks up or price moves far from EMA
            if not downtrend or abs(close[i] - ema_20[i]) / ema_20[i] > 0.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals