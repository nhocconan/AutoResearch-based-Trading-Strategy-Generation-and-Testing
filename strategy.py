#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivotBreakout_1dTrend_Volume"
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
    
    # Get weekly data for pivot calculation (1w timeframe)
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 20:
        return np.zeros(n)
    
    # Weekly high, low, close for pivot calculation
    weekly_high = df_w['high'].values
    weekly_low = df_w['low'].values
    weekly_close = df_w['close'].values
    
    # Calculate weekly pivot and support/resistance levels
    # Standard pivot point: (H + L + C) / 3
    # Resistance 1: (2 * P) - L
    # Support 1: (2 * P) - H
    # Resistance 2: P + (H - L)
    # Support 2: P - (H - L)
    pivot_w = (weekly_high + weekly_low + weekly_close) / 3
    r1_w = (2 * pivot_w) - weekly_low
    s1_w = (2 * pivot_w) - weekly_high
    r2_w = pivot_w + (weekly_high - weekly_low)
    s2_w = pivot_w - (weekly_high - weekly_low)
    
    # Align weekly pivot levels to 6h timeframe
    r1_w_aligned = align_htf_to_ltf(prices, df_w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_w, s1_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_w, r2_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_w, s2_w)
    
    # Get daily data for trend filter
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 20:
        return np.zeros(n)
    
    # Daily EMA(50) for trend filter
    daily_close = df_d['close'].values
    close_d = pd.Series(daily_close)
    ema50_d = close_d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_d_aligned = align_htf_to_ltf(prices, df_d, ema50_d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_w_aligned[i]) or np.isnan(s1_w_aligned[i]) or 
            np.isnan(r2_w_aligned[i]) or np.isnan(s2_w_aligned[i]) or 
            np.isnan(ema50_d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma20[i]
        
        if position == 0:
            # Long: Price breaks above weekly R2 with volume and above daily EMA trend
            if close[i] > r2_w_aligned[i] and vol_ok and close[i] > ema50_d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S2 with volume and below daily EMA trend
            elif close[i] < s2_w_aligned[i] and vol_ok and close[i] < ema50_d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below weekly S1 (reversion to mean)
            if close[i] < s1_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above weekly R1 (reversion to mean)
            if close[i] > r1_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals