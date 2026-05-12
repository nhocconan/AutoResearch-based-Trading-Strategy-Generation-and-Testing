#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Use daily EMA50 for trend filter, and 4h Camarilla R1/S1 levels for breakout entries.
# Long when price breaks above R1 with volume and daily trend up; short when breaks below S1 with volume and daily trend down.
# Exit when price returns to 4h Pivot level. Designed for low frequency (20-40 trades/year) using 1d trend and 4h breakouts.
# Works in bull markets by following uptrend, and in bear markets by following downtrend.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Daily data for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Daily EMA50 for trend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 4h data for Camarilla levels ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for previous 4h bar
    typical_4h = (high_4h + low_4h + close_4h) / 3
    pivot_4h = typical_4h
    range_4h = high_4h - low_4h
    r1_4h = close_4h + (range_4h * 1.1 / 12)
    s1_4h = close_4h - (range_4h * 1.1 / 12)
    
    # Align Camarilla levels to 4h (wait for 4h bar to close)
    pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # Volume confirmation (20-period average on 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(pivot_4h_aligned[i]) or np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above R1 with volume, in uptrend
            if close[i] > r1_4h_aligned[i] and vol_ok and trend_up:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume, in downtrend
            elif close[i] < s1_4h_aligned[i] and vol_ok and trend_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price returns to pivot level or trend reverses
            if close[i] <= pivot_4h_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot level or trend reverses
            if close[i] >= pivot_4h_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals