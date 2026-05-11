#!/usr/bin/env python3
"""
4h_Rolling_Regression_Trend_v1
Hypothesis: Use 1d linear regression slope (trend strength) to filter 4h breakouts.
In bull markets, 1d regression slope > 0.001 (uptrend); in bear markets, slope < -0.001 (downtrend).
Only trade breakouts in the direction of the 1d trend. This avoids counter-trend whipsaws.
Price channel: Donchian(20) breakout. Volume confirmation: volume > 1.5x 20-period average.
Target: 20-50 total trades over 4 years on 4h timeframe.
"""

name = "4h_Rolling_Regression_Trend_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1D Data for Linear Regression Slope (Trend) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for regression
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 20-period linear regression slope on 1d closes
    # Slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - (sum(x))^2)
    # where x = [0,1,2,...,n-1], y = close prices
    def rolling_slope(y, window):
        """Calculate rolling linear regression slope"""
        n = len(y)
        slope = np.full(n, np.nan)
        if window < 2:
            return slope
        
        for i in range(window-1, n):
            x = np.arange(window)
            y_window = y[i-window+1:i+1]
            # Calculate slope using least squares
            x_mean = np.mean(x)
            y_mean = np.mean(y_window)
            numerator = np.sum((x - x_mean) * (y_window - y_mean))
            denominator = np.sum((x - x_mean) ** 2)
            if denominator != 0:
                slope[i] = numerator / denominator
        return slope
    
    slope_20 = rolling_slope(close_1d, 20)
    slope_aligned = align_htf_to_ltf(prices, df_1d, slope_20)
    
    # === 4H Indicators ===
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(slope_aligned[i]) or 
            np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend direction from 1d regression slope
        # Only trade in direction of trend: slope > 0.001 = uptrend, slope < -0.001 = downtrend
        strong_uptrend = slope_aligned[i] > 0.001
        strong_downtrend = slope_aligned[i] < -0.001
        
        # Volume confirmation: volume > 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND strong uptrend AND volume confirmation
            if close[i] > high_20[i] and strong_uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND strong downtrend AND volume confirmation
            elif close[i] < low_20[i] and strong_downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low OR trend weakens
            if close[i] < low_20[i] or not strong_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price breaks above Donchian high OR trend weakens
            if close[i] > high_20[i] or not strong_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals