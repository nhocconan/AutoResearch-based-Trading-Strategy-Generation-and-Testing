#!/usr/bin/env python3
"""
6h_LR_Touch_Trend
Linear regression touch strategy with trend confirmation:
- Linear regression slope (20) determines trend direction
- Long when price touches lower 1.5σ band + positive slope
- Short when price touches upper 1.5σ band + negative slope
- Exit when price crosses back to midline or slope reverses
- Uses 1w trend filter (price > weekly EMA50 for longs, < for shorts)
- Designed for 20-30 trades/year per symbol
Works in bull markets (trend + pullback) and bear markets (trend + bounce)
"""

import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import polyfit
from mtf_data import get_htf_data, align_htf_to_ltf

def linear_regression_slope(values, period):
    """Calculate linear regression slope over given period."""
    n = len(values)
    slope = np.full(n, np.nan)
    for i in range(period-1, n):
        y = values[i-period+1:i+1]
        x = np.arange(len(y))
        try:
            b, _ = polyfit(x, y, 1)
            slope[i] = b
        except:
            slope[i] = np.nan
    return slope

def linear_regression_intercept(values, period):
    """Calculate linear regression intercept over given period."""
    n = len(values)
    intercept = np.full(n, np.nan)
    for i in range(period-1, n):
        y = values[i-period+1:i+1]
        x = np.arange(len(y))
        try:
            b, a = polyfit(x, y, 1)
            intercept[i] = a
        except:
            intercept[i] = np.nan
    return intercept

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Linear regression parameters
    lr_period = 20
    multi = 1.5
    
    # Calculate linear regression slope and intercept
    lr_slope = linear_regression_slope(close, lr_period)
    lr_intercept = linear_regression_intercept(close, lr_period)
    
    # Calculate linear regression values and standard deviation
    lr_value = np.full(n, np.nan)
    std_dev = np.full(n, np.nan)
    for i in range(lr_period-1, n):
        x = np.arange(lr_period)
        y = close[i-lr_period+1:i+1]
        try:
            b, a = polyfit(x, y, 1)
            lr_vals = a + b * x
            lr_value[i] = a + b * (lr_period - 1)  # current value
            std_dev[i] = np.std(y - lr_vals)
        except:
            lr_value[i] = np.nan
            std_dev[i] = np.nan
    
    # Calculate upper and lower bands
    upper_band = lr_value + multi * std_dev
    lower_band = lr_value - multi * std_dev
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50
    if len(close_1w) >= 50:
        ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    else:
        ema_50_1w = np.full(len(close_1w), np.nan)
    
    # Align weekly EMA50 to 6h timeframe
    ema_50_1w_6h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = lr_period  # need sufficient data for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(lr_slope[i]) or np.isnan(lr_value[i]) or 
            np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(ema_50_1w_6h[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from weekly EMA50
        uptrend = close[i] > ema_50_1w_6h[i]
        downtrend = close[i] < ema_50_1w_6h[i]
        
        # Price touching bands
        touch_lower = low[i] <= lower_band[i]
        touch_upper = high[i] >= upper_band[i]
        
        # Price at midline (exit condition)
        at_midline = abs(close[i] - lr_value[i]) < 0.1 * std_dev[i]
        
        if position == 0:
            # Long: touch lower band + uptrend
            if touch_lower and uptrend and lr_slope[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: touch upper band + downtrend
            elif touch_upper and downtrend and lr_slope[i] < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price at midline or slope turns negative
            if at_midline or lr_slope[i] < 0:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price at midline or slope turns positive
            if at_midline or lr_slope[i] > 0:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_LR_Touch_Trend"
timeframe = "6h"
leverage = 1.0