#!/usr/bin/env python3
"""
6h_Linear_Regression_Channel_v1
Hypothesis: Price tends to revert to the mean within a 60-period linear regression channel on 6h timeframe, 
with entries at ±1.5 standard deviations from the regression line. Uses 1-week trend filter to align with 
higher timeframe momentum and volume confirmation to avoid false signals. Designed to work in both bull 
and bear markets by fading extremes in ranging conditions and following trend when channel breaks.
Target: 20-40 trades/year to minimize fee drag.
"""

name = "6h_Linear_Regression_Channel_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 50:
        return np.zeros(n)
    
    close_w = df_w['close'].values
    
    # Calculate 50-period EMA on weekly for trend filter
    ema50_w = pd.Series(close_w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_w_aligned = align_htf_to_ltf(prices, df_w, ema50_w)
    
    # Get 6h data for linear regression channel and volume
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 60:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate 60-period linear regression and standard error on 6h close
    def linreg(src, length):
        if length < 2:
            return np.full_like(src, np.nan), np.full_like(src, np.nan)
        sum_x = (length - 1) * length / 2.0
        sum_x2 = (length - 1) * length * (2 * length - 1) / 6.0
        src_series = pd.Series(src)
        sum_y = src_series.rolling(window=length, min_periods=length).sum().values
        sum_xy = src_series.rolling(window=length, min_periods=length).apply(
            lambda x: np.sum(x * np.arange(length)), raw=True
        ).values
        slope = (length * sum_xy - sum_x * sum_y) / (length * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / length
        return slope * np.arange(length) + intercept, slope
    
    # Calculate linear regression values and standard error
    lr_slope = np.full(n, np.nan)
    lr_intercept = np.full(n, np.nan)
    lr_value = np.full(n, np.nan)
    std_err = np.full(n, np.nan)
    
    for i in range(59, n):  # Start from index 59 to have 60 points (0-59)
        window_close = close_6h[i-59:i+1]
        if len(window_close) < 60:
            continue
        sum_x = 60 * 59 / 2.0  # sum of 0 to 59
        sum_x2 = 60 * 59 * 119 / 6.0  # sum of squares of 0 to 59
        sum_y = np.sum(window_close)
        sum_xy = np.sum(window_close * np.arange(60))
        slope = (60 * sum_xy - sum_x * sum_y) / (60 * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / 60
        lr_slope[i] = slope
        lr_intercept[i] = intercept
        lr_value[i] = slope * 59 + intercept  # Value at bar i (most recent)
        # Standard error of estimate
        y_pred = slope * np.arange(60) + intercept
        residuals = window_close - y_pred
        std_err[i] = np.sqrt(np.sum(residuals**2) / 58)  # 60-2 degrees of freedom
    
    # Align LR values to 6h timeframe (already calculated on 6h data)
    lr_value_aligned = lr_value  # Already on 6h timeframe
    std_err_aligned = std_err    # Already on 6h timeframe
    
    # Calculate upper and lower channel lines (±1.5 standard errors)
    upper_channel = lr_value_aligned + 1.5 * std_err_aligned
    lower_channel = lr_value_aligned - 1.5 * std_err_aligned
    
    # Volume filter: current 6h volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume_6h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema20_aligned = align_htf_to_ltf(prices, df_6h, vol_ema20)
    volume_filter = volume_6h > vol_ema20_aligned * 1.5
    
    # 6h data for signal generation
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly trend (50), 60-period LR (59)
    start_idx = max(50, 59)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_w_aligned[i]) or 
            np.isnan(lr_value_aligned[i]) or
            np.isnan(std_err_aligned[i]) or
            np.isnan(upper_channel[i]) or
            np.isnan(lower_channel[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: weekly EMA50 direction
        # Use slope of weekly EMA to determine trend (avoid look-ahead)
        if i >= 51:  # Need at least 2 points to calculate slope
            ema50_prev = ema50_w_aligned[i-1]
            ema50_curr = ema50_w_aligned[i]
            w_trend_up = ema50_curr > ema50_prev
            w_trend_down = ema50_curr < ema50_prev
        else:
            w_trend_up = False
            w_trend_down = False
        
        # Volume filter
        vol_filter = volume[i] > vol_ema20_aligned[i] if i < len(vol_ema20_aligned) else False
        
        if position == 0:
            # Long entry: price at or below lower channel + up trend + volume
            if low[i] <= lower_channel[i] and w_trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price at or above upper channel + down trend + volume
            elif high[i] >= upper_channel[i] and w_trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches middle line or trend fails
            if close[i] >= lr_value_aligned[i] or not w_trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches middle line or trend fails
            if close[i] <= lr_value_aligned[i] or not w_trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals