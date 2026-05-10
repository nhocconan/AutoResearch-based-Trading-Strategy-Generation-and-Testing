#!/usr/bin/env python3
"""
6h_Angle_Based_Trend_Follow
Hypothesis: On 6h timeframe, price angle (slope) over 20 periods predicts medium-term trend.
Price above/below 20-period linear regression slope + intercept acts as dynamic support/resistance.
Combined with 1d EMA trend filter and volume confirmation to filter false signals.
Works in bull/bear by following trend only when aligned with higher timeframe.
Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
"""

name = "6h_Angle_Based_Trend_Follow"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from scipy import stats
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 60-period linear regression slope and intercept on 6b (for 6h timeframe)
    # Using 20 periods for regression window
    lookback = 20
    slopes = np.full(n, np.nan)
    intercepts = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        y = close[i-lookback+1:i+1]
        x = np.arange(lookback)
        slope, intercept, _, _, _ = stats.linregress(x, y)
        slopes[i] = slope
        intercepts[i] = intercept
    
    # Calculate regression value (predicted close) at current point
    # This gives us the dynamic support/resistance level
    reg_value = slopes * (lookback-1) + intercepts  # value at end of window
    
    # Calculate 1d EMA20 for trend filter
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume confirmation (20-period MA on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need regression (20) and volume MA (20)
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(reg_value[i]) or 
            np.isnan(ema_20_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Higher timeframe trend filter
        uptrend_1d = close[i] > ema_20_1d_aligned[i]
        downtrend_1d = close[i] < ema_20_1d_aligned[i]
        
        # Volume confirmation (>1.5x average volume)
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: uptrend + price above regression line + volume confirmation
            if uptrend_1d and close[i] > reg_value[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + price below regression line + volume confirmation
            elif downtrend_1d and close[i] < reg_value[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price crosses below regression line
            if not uptrend_1d or close[i] < reg_value[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price crosses above regression line
            if not downtrend_1d or close[i] > reg_value[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals