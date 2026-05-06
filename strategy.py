#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly pivot levels (S1/S2/R1/R2) with volume confirmation and 1d EMA200 trend filter
# - Long when price crosses above weekly S2 with volume spike and price above 1d EMA200
# - Short when price crosses below weekly R2 with volume spike and price below 1d EMA200
# - Exit when price crosses opposite pivot level (S1 for long, R1 for short)
# - Weekly pivots provide stronger support/resistance than daily, reducing false signals
# - EMA200 filter ensures alignment with long-term trend, improving performance in both bull and bear markets
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "6h_WeeklyPivot_S2R2_1dEMA200_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's data)
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Weekly pivot point: P = (H + L + C) / 3
    pivot_w = (high_w + low_w + close_w) / 3
    range_w = high_w - low_w
    
    # Weekly support/resistance levels (S1, S2, R1, R2)
    s1_w = 2 * pivot_w - high_w
    s2_w = pivot_w - range_w
    r1_w = 2 * pivot_w - low_w
    r2_w = pivot_w + range_w
    
    # Get daily data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA200
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly and daily indicators to 6h timeframe
    s2_w_6h = align_htf_to_ltf(prices, df_w, s2_w)
    r2_w_6h = align_htf_to_ltf(prices, df_w, r2_w)
    s1_w_6h = align_htf_to_ltf(prices, df_w, s1_w)
    r1_w_6h = align_htf_to_ltf(prices, df_w, r1_w)
    ema_200_1d_6h = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume filters (6h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(s2_w_6h[i]) or np.isnan(r2_w_6h[i]) or 
            np.isnan(s1_w_6h[i]) or np.isnan(r1_w_6h[i]) or 
            np.isnan(ema_200_1d_6h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above S2 with volume spike and above EMA200
            if close[i] > s2_w_6h[i] and close[i-1] <= s2_w_6h[i-1] and volume_spike[i] and close[i] > ema_200_1d_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below R2 with volume spike and below EMA200
            elif close[i] < r2_w_6h[i] and close[i-1] >= r2_w_6h[i-1] and volume_spike[i] and close[i] < ema_200_1d_6h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below S1
            if close[i] < s1_w_6h[i] and close[i-1] >= s1_w_6h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above R1
            if close[i] > r1_w_6h[i] and close[i-1] <= r1_w_6h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals