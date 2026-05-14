#!/usr/bin/env python3
# 4H_1D_Camarilla_R2_S2_Breakout_1dEMA50_Trend_Volume
# Hypothesis: Use daily EMA50 for trend filter with daily Camarilla R2/S2 breakouts.
# R2/S2 levels (at 1.1/2 of range) provide stronger breakout signals than R1/S1.
# Daily EMA50 offers robust trend filtering suitable for institutional timeframes.
# Volume confirmation ensures breakouts have conviction. Works in bull/bear via trend filter.
# Target: 75-200 total trades over 4 years (19-50/year).

name = "4H_1D_Camarilla_R2_S2_Breakout_1dEMA50_Trend_Volume"
timeframe = "4h"
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
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and Camarilla levels (R2, S2)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_ = high_1d - low_1d
    r2 = pivot + range_ * 1.1 / 2  # R2 = pivot + (range * 1.1 / 2)
    s2 = pivot - range_ * 1.1 / 2  # S2 = pivot - (range * 1.1 / 2)
    
    # Get daily data for EMA50 trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 4h
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R2 + above daily EMA50 + volume confirmation
            if close[i] > r2_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S2 + below daily EMA50 + volume confirmation
            elif close[i] < s2_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below daily EMA50 (trend change)
            if close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above daily EMA50 (trend change)
            if close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals