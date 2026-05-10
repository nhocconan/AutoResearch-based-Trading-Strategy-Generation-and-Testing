#!/usr/bin/env python3
# 12H_WedgeBreakout_With_1wTrend_Volume
# Hypothesis: Breakouts from ascending/descending wedges (defined by higher lows/lower highs)
# combined with 1-week trend and volume confirmation. Works in bull/bear by following 1w trend.
# Uses 12h timeframe to limit trades (<30/year) and avoid fee drag.

name = "12H_WedgeBreakout_With_1wTrend_Volume"
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
    
    # Wedge detection: higher lows and lower highs over 5 periods
    # Higher low: low[i] > low[i-5]
    # Lower high: high[i] < high[i-5]
    higher_low = np.zeros(n, dtype=bool)
    lower_high = np.zeros(n, dtype=bool)
    
    for i in range(5, n):
        higher_low[i] = low[i] > low[i-5]
        lower_high[i] = high[i] < high[i-5]
    
    # Ascending wedge: higher lows + lower highs (bullish breakout potential)
    # Descending wedge: lower highs + higher lows (same condition, context by trend)
    wedge_forming = higher_low & lower_high
    
    # Volume confirmation: 2x average volume
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # 1-week trend (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_up = close_1w > ema50_1w
    trend_1w_down = close_1w < ema50_1w
    
    # Align 1w trend to 12h
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or
            np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 2.0
        
        wedge_now = wedge_forming[i]
        trend_up = trend_1w_up_aligned[i] > 0.5
        trend_down = trend_1w_down_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: wedge breakout up + 1w uptrend + volume
            # Breakout up: close above previous high
            if i > 0 and wedge_now and close[i] > high[i-1] and trend_up and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: wedge breakout down + 1w downtrend + volume
            # Breakout down: close below previous low
            elif i > 0 and wedge_now and close[i] < low[i-1] and trend_down and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when wedge breaks down (close below wedge support)
            # Simple exit: close below previous low (trailing stop logic)
            if i > 0 and close[i] < low[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when wedge breaks up (close above wedge resistance)
            if i > 0 and close[i] > high[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals