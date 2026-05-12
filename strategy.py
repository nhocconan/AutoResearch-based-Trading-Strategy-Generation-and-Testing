#!/usr/bin/env python3
"""
6h_Pivots_Triangle_Trend_With_Volume
For 6h timeframe, uses 12h pivot points and triangle pattern recognition:
- Long when price breaks above upper triangle trendline with 12h uptrend and volume spike
- Short when price breaks below lower triangle trendline with 12h downtrend and volume spike
Uses 12h high/low to draw trendlines connecting last 3 swing points
Designed for low trade frequency (~50-150 total trades over 4 years) to minimize fee drift
Works in bull/bear markets by following 12h trend while using 6s triangle breakouts for entries
"""

name = "6h_Pivots_Triangle_Trend_With_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: >2x 20-period average (on 6h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 12h data for pivot points and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h pivot points (standard)
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    r1_12h = 2 * pivot_12h - low_12h
    s1_12h = 2 * pivot_12h - high_12h
    
    # 12h EMA20 for trend filter
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Swing points detection on 12h (for triangle trendlines)
    # Swing high: high > previous 2 and next 2
    # Swing low: low < previous 2 and next 2
    swing_high = np.zeros(len(high_12h), dtype=bool)
    swing_low = np.zeros(len(low_12h), dtype=bool)
    
    for i in range(2, len(high_12h) - 2):
        if (high_12h[i] > high_12h[i-1] and high_12h[i] > high_12h[i-2] and
            high_12h[i] > high_12h[i+1] and high_12h[i] > high_12h[i+2]):
            swing_high[i] = True
        if (low_12h[i] < low_12h[i-1] and low_12h[i] < low_12h[i-2] and
            low_12h[i] < low_12h[i+1] and low_12h[i] < low_12h[i+2]):
            swing_low[i] = True
    
    # Get last 3 swing points for trendlines
    def get_last_n_swings(arr, n=3):
        indices = np.where(arr)[0]
        if len(indices) < n:
            return indices[-len(indices):] if len(indices) > 0 else np.array([])
        return indices[-n:]
    
    last_swing_high_idx = get_last_n_swings(swing_high, 3)
    last_swing_low_idx = get_last_n_swings(swing_low, 3)
    
    # Calculate trendlines
    # Upper trendline: connect last 3 swing highs
    # Lower trendline: connect last 3 swing lows
    upper_slope = np.zeros(len(high_12h))
    upper_intercept = np.zeros(len(high_12h))
    lower_slope = np.zeros(len(low_12h))
    lower_intercept = np.zeros(len(low_12h))
    
    if len(last_swing_high_idx) >= 2:
        # Linear regression for upper trendline
        x = last_swing_high_idx.astype(float)
        y = high_12h[last_swing_high_idx]
        if len(x) >= 2:
            slope = np.polyfit(x, y, 1)[0]
            intercept = np.polyfit(x, y, 1)[1]
            upper_slope[:] = slope
            upper_intercept[:] = intercept
    
    if len(last_swing_low_idx) >= 2:
        # Linear regression for lower trendline
        x = last_swing_low_idx.astype(float)
        y = low_12h[last_swing_low_idx]
        if len(x) >= 2:
            slope = np.polyfit(x, y, 1)[0]
            intercept = np.polyfit(x, y, 1)[1]
            lower_slope[:] = slope
            lower_intercept[:] = intercept
    
    # Align all indicators to 6h timeframe
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    upper_slope_aligned = align_htf_to_ltf(prices, df_12h, upper_slope)
    upper_intercept_aligned = align_htf_to_ltf(prices, df_12h, upper_intercept)
    lower_slope_aligned = align_htf_to_ltf(prices, df_12h, lower_slope)
    lower_intercept_aligned = align_htf_to_ltf(prices, df_12h, lower_intercept)
    
    # Calculate current trendline values
    # Need current bar index in 12h terms
    bars_per_12h = 2  # 2 x 6h bars in 12h
    current_12h_idx = np.arange(len(prices)) // bars_per_12h
    # Ensure we don't go out of bounds
    current_12h_idx = np.minimum(current_12h_idx, len(df_12h) - 1)
    
    upper_trendline = upper_slope_aligned * current_12h_idx + upper_intercept_aligned
    lower_trendline = lower_slope_aligned * current_12h_idx + lower_intercept_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(pivot_12h_aligned[i]) or
            np.isnan(r1_12h_aligned[i]) or
            np.isnan(s1_12h_aligned[i]) or
            np.isnan(ema_20_12h_aligned[i]) or
            np.isnan(upper_trendline[i]) or
            np.isnan(lower_trendline[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper trendline + 12h EMA20 uptrend + volume spike
            # Additional filter: price above pivot (bullish bias)
            if (close[i] > upper_trendline[i] and 
                close[i] > ema_20_12h_aligned[i] and
                close[i] > pivot_12h_aligned[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower trendline + 12h EMA20 downtrend + volume spike
            # Additional filter: price below pivot (bearish bias)
            elif (close[i] < lower_trendline[i] and 
                  close[i] < ema_20_12h_aligned[i] and
                  close[i] < pivot_12h_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below lower trendline OR closes below 12h EMA20
            if (close[i] < lower_trendline[i]) or \
               (close[i] < ema_20_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above upper trendline OR closes above 12h EMA20
            if (close[i] > upper_trendline[i]) or \
               (close[i] > ema_20_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals