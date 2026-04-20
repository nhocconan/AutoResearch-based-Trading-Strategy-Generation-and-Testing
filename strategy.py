#!/usr/bin/env python3
"""
12h_1d_1w_RollingRegressionTrend_With_Volume_Filter_v1
Concept: 12h timeframe with linear regression slope as trend filter, volume confirmation, and multi-timeframe alignment.
- Uses 12h linear regression slope (20-period) for trend direction
- Confirms with 1d trend (linear regression slope) to avoid counter-trend trades
- Requires volume > 1.5x 20-period average for entry
- Exits when trend reverses or volume drops
- Designed for low trade frequency (~15-30/year) to minimize fee drag
- Works in bull/bear: regression adapts to slope, volume confirms institutional interest
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_RollingRegressionTrend_With_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 10:
        return np.zeros(n)
    
    # === 12h: Linear regression slope (20-period) for trend ===
    close = prices['close'].values
    def rolling_linreg_slope(arr, window):
        slopes = np.full_like(arr, np.nan, dtype=np.float64)
        for i in range(window - 1, len(arr)):
            y = arr[i - window + 1:i + 1]
            x = np.arange(window)
            if np.all(np.isnan(y)) or np.std(y) == 0:
                slopes[i] = 0
            else:
                slope = np.polyfit(x, y, 1)[0]
                slopes[i] = slope
        return slopes
    
    slope_12h = rolling_linreg_slope(close, 20)
    
    # === 1d: Linear regression slope (20-period) for higher timeframe trend ===
    close_1d = df_1d['close'].values
    slope_1d = rolling_linreg_slope(close_1d, 20)
    slope_1d_aligned = align_htf_to_ltf(prices, df_1d, slope_1d)
    
    # === 1w: Linear regression slope (10-period) for major trend filter ===
    close_1w = df_1w['close'].values
    slope_1w = rolling_linreg_slope(close_1w, 10)
    slope_1w_aligned = align_htf_to_ltf(prices, df_1w, slope_1w)
    
    # === 12h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        slope_12h_val = slope_12h[i]
        slope_1d_val = slope_1d_aligned[i]
        slope_1w_val = slope_1w_aligned[i]
        close_val = close[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(slope_12h_val) or np.isnan(slope_1d_val) or np.isnan(slope_1w_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Positive slope on all timeframes with volume confirmation
            if slope_12h_val > 0 and slope_1d_val > 0 and slope_1w_val > 0 and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Negative slope on all timeframes with volume confirmation
            elif slope_12h_val < 0 and slope_1d_val < 0 and slope_1w_val < 0 and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Trend turns negative on any timeframe or volume drops
            if slope_12h_val <= 0 or slope_1d_val <= 0 or slope_1w_val <= 0 or vol_ratio_val < 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Trend turns positive on any timeframe or volume drops
            if slope_12h_val >= 0 or slope_1d_val >= 0 or slope_1w_val >= 0 or vol_ratio_val < 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals