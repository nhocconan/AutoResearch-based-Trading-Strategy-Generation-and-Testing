#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot R4/S4 breakout with 4h EMA50 trend filter and volume confirmation
# Long when price breaks above 1d Camarilla R4 level AND 4h EMA50 > EMA50 previous (uptrend) AND volume > 1.8 * avg_volume(20) on 4h
# Short when price breaks below 1d Camarilla S4 level AND 4h EMA50 < EMA50 previous (downtrend) AND volume > 1.8 * avg_volume(20) on 4h
# Exit when price retests the 1d Camarilla pivot point (midpoint of R4/S4)
# Uses discrete sizing 0.25 to reduce fee churn and manage drawdown
# Target: 80-150 total trades over 4 years (20-37/year) for 4h timeframe
# 1d Camarilla R4/S4 provides stronger breakout levels than R3/S3 with higher probability follow-through
# 4h EMA50 ensures we trade with the intermediate trend filter
# Volume confirmation validates breakout strength while limiting false signals
# Works in both bull (buy breakouts) and bear (sell breakdowns) markets by trading with the 4h trend

name = "4h_1dCamarilla_R4S4_Breakout_4hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least 5 completed daily bars for pivot calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (R4, S4, and pivot point)
    # Camarilla formulas:
    # Pivot = (H + L + C) / 3
    # R4 = C + (H - L) * 1.1 / 2
    # S4 = C - (H - L) * 1.1 / 2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r4_1d = close_1d + (high_1d - low_1d) * 1.1 / 2.0
    s4_1d = close_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align 1d Camarilla levels to 4h timeframe (wait for completed 1d bar)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Get 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need at least 50 completed 4h bars for EMA50
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate volume confirmation: volume > 1.8 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R4, 4h EMA50 > EMA50 previous (uptrend), volume spike
            if (close[i] > r4_aligned[i] and 
                ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla S4, 4h EMA50 < EMA50 previous (downtrend), volume spike
            elif (close[i] < s4_aligned[i] and 
                  ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests the 1d Camarilla pivot point
            if close[i] <= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests the 1d Camarilla pivot point
            if close[i] >= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals