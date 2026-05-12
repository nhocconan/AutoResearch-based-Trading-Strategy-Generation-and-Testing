#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
# Hypothesis: Use Camarilla R1/S1 levels on 4h as breakout levels, with 4h trend filter (EMA50) and volume confirmation.
# Enter long when price breaks above R1 with volume, short when breaks below S1 with volume, only in direction of 4h trend.
# Exit when price returns to Camarilla Pivot level or trend reverses.
# Designed for low frequency (15-30 trades/year) by using 4h for signal direction and 1h only for entry timing.
# Works in both bull and bear markets by following higher timeframe trend.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
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
    
    # === 4h data for Camarilla levels and trend ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for previous 4h bar
    # Typical price = (high + low + close) / 3
    typical_4h = (high_4h + low_4h + close_4h) / 3
    # Pivot = typical price
    pivot_4h = typical_4h
    # Range = high - low
    range_4h = high_4h - low_4h
    # R1 = close + (range * 1.1 / 12)
    r1_4h = close_4h + (range_4h * 1.1 / 12)
    # S1 = close - (range * 1.1 / 12)
    s1_4h = close_4h - (range_4h * 1.1 / 12)
    
    # Align Camarilla levels to 1h (wait for 4h bar to close)
    pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation (24-period average on 1h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(pivot_4h_aligned[i]) or np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 4h EMA50
        trend_up = close[i] > ema_50_4h_aligned[i]
        trend_down = close[i] < ema_50_4h_aligned[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_24[i]
        
        if position == 0:
            # LONG: Price breaks above R1 with volume, in uptrend
            if close[i] > r1_4h_aligned[i] and vol_ok and trend_up:
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 with volume, in downtrend
            elif close[i] < s1_4h_aligned[i] and vol_ok and trend_down:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # EXIT LONG: Price returns to pivot level or trend reverses
            if close[i] <= pivot_4h_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price returns to pivot level or trend reverses
            if close[i] >= pivot_4h_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals