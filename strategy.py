#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_4hTrend_1dVolSlope
# Hypothesis: Combines 4h trend filter with 1d volume slope to reduce false breakouts in 1h timeframe.
# Uses 1d Camarilla R1/S1 for entry levels, filtered by 4h EMA20 trend and 1d volume slope (rising volume = institutional interest).
# In bull markets: 4h trend up + price breaks R1 with rising 1d volume = long continuation.
# In bear markets: 4h trend down + price breaks S1 with rising 1d volume = short continuation.
# The 4h trend filter reduces whipsaws vs 1d, improving robustness in both regimes.
# Volume slope filter ensures breakouts are supported by increasing participation.
# Target: 20-30 trades/year to minimize fee drag while maintaining edge.

name = "1h_Camarilla_R1_S1_4hTrend_1dVolSlope"
timeframe = "1h"
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
    
    # Get 1d data for Camarilla pivot calculation (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels: R1, S1
    camarilla_range = high_1d - low_1d
    r1 = close_1d + 1.1 * camarilla_range / 12
    s1 = close_1d - 1.1 * camarilla_range / 12
    
    # Get 4h data for trend filter (EMA20)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Get 1d volume slope (5-period linear regression slope)
    vol_1d = df_1d['volume'].values
    vol_slope_1d = np.zeros(len(vol_1d))
    for i in range(4, len(vol_1d)):
        y = vol_1d[i-4:i+1]
        x = np.arange(5)
        slope = np.polyfit(x, y, 1)[0]
        vol_slope_1d[i] = slope
    
    # Align all indicators to 1h timeframe
    r1_1h = align_htf_to_ltf(prices, df_1d, r1)
    s1_1h = align_htf_to_ltf(prices, df_1d, s1)
    ema_20_4h_1h = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    vol_slope_1d_1h = align_htf_to_ltf(prices, df_1d, vol_slope_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_1h[i]) or np.isnan(s1_1h[i]) or 
            np.isnan(ema_20_4h_1h[i]) or np.isnan(vol_slope_1d_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price > R1, above 4h EMA20 trend, positive 1d volume slope
            if close[i] > r1_1h[i] and close[i] > ema_20_4h_1h[i] and vol_slope_1d_1h[i] > 0:
                signals[i] = 0.20
                position = 1
            # Short: Price < S1, below 4h EMA20 trend, positive 1d volume slope
            elif close[i] < s1_1h[i] and close[i] < ema_20_4h_1h[i] and vol_slope_1d_1h[i] > 0:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: Price < R1 or below 4h EMA20 trend
            if close[i] < r1_1h[i] or close[i] < ema_20_4h_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: Price > S1 or above 4h EMA20 trend
            if close[i] > s1_1h[i] or close[i] > ema_20_4h_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals