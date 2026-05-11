#!/usr/bin/env python3
"""
6h_12h_Camarilla_R3S3_Breakout_1dTrend_Volume
Hypothesis: At 6h timeframe, breakouts from Camarilla R3/S3 levels with 1d trend filter (EMA34) and volume confirmation capture institutional breakout moves. Works in bull markets via breakout continuation and in bear markets via mean-reversion failures at R3/S3 (fading false breakouts). Target: 15-35 trades/year (60-140 total over 4 years).
"""
name = "6h_12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for Camarilla pivot calculation (prior day's OHLC)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 12h Camarilla levels (using prior 12h bar's OHLC) ---
    # For each 6h bar, we need the Camarilla levels based on the prior completed 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate pivot and ranges for prior 12h bar
    pivot_12h = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    
    # Camarilla levels
    r3_12h = pivot_12h + range_12h * 1.1 / 4
    s3_12h = pivot_12h - range_12h * 1.1 / 4
    r4_12h = pivot_12h + range_12h * 1.1 / 2
    s4_12h = pivot_12h - range_12h * 1.1 / 2
    
    # --- 1d EMA34 trend ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # EMA34 with proper initialization
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema_1d[33] = np.mean(close_1d[0:34])  # Simple average for first EMA value
        for i in range(34, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 / (34 + 1)) + (ema_1d[i-1] * (33 / (34 + 1)))
    
    # EMA slope for trend direction
    ema_slope_1d = np.full(len(close_1d), np.nan)
    for i in range(35, len(close_1d)):
        ema_slope_1d[i] = ema_1d[i] - ema_1d[i-1]
    
    # --- 6h volume MA(20) for volume confirmation ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # --- Align all HTF indicators to 6h timeframe ---
    # Align Camarilla levels (based on prior 12h bar)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # Align 1d EMA and slope
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    ema_slope_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_slope_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 12h data (2 bars), 1d EMA34 (34 bars), volume MA20 (20 bars)
    start_idx = max(2, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or
            np.isnan(r4_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(ema_slope_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > vol_ma[i] * 1.5
        
        if position == 0:
            # Long breakout: price breaks above R3 with 1d uptrend and volume
            if close[i] > r3_12h_aligned[i] and ema_slope_1d_aligned[i] > 0 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below S3 with 1d downtrend and volume
            elif close[i] < s3_12h_aligned[i] and ema_slope_1d_aligned[i] < 0 and vol_spike:
                signals[i] = -0.25
                position = -1
            # Fade at R4/S4: mean reversion at extreme levels (works in ranging/bear markets)
            elif close[i] > r4_12h_aligned[i] and ema_slope_1d_aligned[i] < 0 and vol_spike:
                signals[i] = -0.20  # Fade the breakout at R4
                position = -1
            elif close[i] < s4_12h_aligned[i] and ema_slope_1d_aligned[i] > 0 and vol_spike:
                signals[i] = 0.20   # Fade the breakout at S4
                position = 1
        else:
            if position == 1:
                # Exit long: price breaks below S3 OR 1d trend turns down
                if close[i] < s3_12h_aligned[i] or ema_slope_1d_aligned[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above R3 OR 1d trend turns up
                if close[i] > r3_12h_aligned[i] or ema_slope_1d_aligned[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals