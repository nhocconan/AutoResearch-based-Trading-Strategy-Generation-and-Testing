#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_4hTrend_1dVolSlope
# Hypothesis: 1h Camarilla pivot breakout at R1/S1 with 4h EMA trend filter and 1d volume slope confirmation.
# Uses daily volume slope to detect institutional accumulation/distribution, works in bull/bear via trend filter.
# Targets 20-50 trades/year by requiring confluence of pivot break, trend alignment, and volume slope.

name = "1h_Camarilla_R1_S1_4hTrend_1dVolSlope"
timeframe = "1h"
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
    
    # Get 4h data for Camarilla pivot calculation and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate previous 4h bar's Camarilla levels (R1, S1)
    # Using previous bar to avoid look-ahead
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = pivot + (prev_high - prev_low) * 1.1 / 12.0
    s1 = pivot - (prev_high - prev_low) * 1.1 / 12.0
    
    # Calculate 4h EMA20 for trend filter
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Get 1d data for volume slope confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    # Calculate volume slope (5-period linear regression slope)
    def linreg_slope(arr, window):
        if len(arr) < window:
            return np.full_like(arr, np.nan)
        slopes = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            y = arr[i-window+1:i+1]
            x = np.arange(window)
            if np.all(np.isnan(y)):
                slopes[i] = np.nan
            else:
                # Remove NaNs if any
                valid = ~np.isnan(y)
                if np.sum(valid) < 2:
                    slopes[i] = np.nan
                else:
                    x_valid = x[valid]
                    y_valid = y[valid]
                    slope = np.polyfit(x_valid, y_valid, 1)[0]
                    slopes[i] = slope
        return slopes
    
    vol_slope_1d = linreg_slope(volume_1d, 5)
    
    # Align all indicators to 1h timeframe
    r1_1h = align_htf_to_ltf(prices, df_4h, r1)
    s1_1h = align_htf_to_ltf(prices, df_4h, s1)
    ema_20_4h_1h = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    vol_slope_1d_1h = align_htf_to_ltf(prices, df_1d, vol_slope_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(r1_1h[i]) or np.isnan(s1_1h[i]) or 
            np.isnan(ema_20_4h_1h[i]) or np.isnan(vol_slope_1d_1h[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > R1 + above 4h EMA20 + positive volume slope
            if close[i] > r1_1h[i] and close[i] > ema_20_4h_1h[i] and vol_slope_1d_1h[i] > 0:
                signals[i] = 0.20
                position = 1
            # Short: Close < S1 + below 4h EMA20 + negative volume slope
            elif close[i] < s1_1h[i] and close[i] < ema_20_4h_1h[i] and vol_slope_1d_1h[i] < 0:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: Close < S1 or below 4h EMA20
            if close[i] < s1_1h[i] or close[i] < ema_20_4h_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: Close > R1 or above 4h EMA20
            if close[i] > r1_1h[i] or close[i] > ema_20_4h_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals