#!/usr/bin/env python3
# 1d_1w_KAMA_Trend_With_Weekly_Volume_Filter
# Hypothesis: Uses daily KAMA direction with weekly volume surge filter to capture trending moves.
# Long when daily KAMA rising and weekly volume > 1.5x average; short when daily KAMA falling and weekly volume > 1.5x average.
# KAMA adapts to market noise, reducing whipsaws in sideways markets. Weekly volume filter ensures participation in institutional moves.
# Works in bull markets (riding uptrends) and bear markets (riding downtrends) by following adaptive trend.
# Weekly timeframe reduces noise and false signals, keeping trade frequency low to avoid fee drag.

name = "1d_1w_KAMA_Trend_With_Weekly_Volume_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for volume filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Daily price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily KAMA (adaptive trend) ---
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=1))
    change = np.insert(change, 0, 0)  # align length
    
    # Volatility sum over 10 periods
    vol = np.abs(np.diff(close, n=1))
    vol = np.insert(vol, 0, 0)
    volatility = np.zeros_like(close)
    for i in range(len(volatility)):
        if i < 10:
            volatility[i] = np.nan
        else:
            volatility[i] = np.sum(vol[i-9:i+1])
    
    # Efficiency Ratio
    er = np.zeros_like(close)
    er[:] = np.nan
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    
    # Smoothing constants
    sc = np.zeros_like(close)
    sc[:] = np.nan
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = er * (fast_sc - slow_sc) + slow_sc
    sc = sc * sc  # square for smoothing
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[:] = np.nan
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(sc[i]) or np.isnan(kama[i-1]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA direction (slope)
    kama_slope = kama - np.roll(kama, 1)
    kama_slope[0] = 0
    # Smooth the slope to reduce noise
    kama_slope_smooth = pd.Series(kama_slope).ewm(span=5, adjust=False, min_periods=1).mean().values
    
    # --- Weekly volume filter ---
    vol_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(vol_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_surge_1w = vol_1w > (vol_ma_1w * 1.5)  # 1.5x average volume
    vol_surge_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_surge_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for KAMA (need ~30 periods for stability)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama_slope_smooth[i]) or
            np.isnan(vol_surge_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction from smoothed KAMA slope
        uptrend = kama_slope_smooth[i] > 0
        downtrend = kama_slope_smooth[i] < 0
        
        if position == 0:
            if uptrend and vol_surge_1w_aligned[i]:
                # Long: rising KAMA + weekly volume surge
                signals[i] = 0.25
                position = 1
            elif downtrend and vol_surge_1w_aligned[i]:
                # Short: falling KAMA + weekly volume surge
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: KAMA slope turns down OR weekly volume surge ends
                if not uptrend or not vol_surge_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: KAMA slope turns up OR weekly volume surge ends
                if not downtrend or not vol_surge_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals