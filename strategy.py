#!/usr/bin/env python3
name = "6h_WeeklyPivot_Trend_DoubleConfirm"
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
    
    # Get weekly data for pivot points (using weekly high/low/close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week's OHLC
    # Standard pivot: P = (H + L + C) / 3
    # Support 1: S1 = (2*P) - H
    # Resistance 1: R1 = (2*P) - L
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = (2 * pivot) - high_1w
    s1 = (2 * pivot) - low_1w
    
    # Align weekly pivot levels to 6h timeframe (using previous week's values)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    
    # 6-day EMA for trend filter (using daily data to avoid whipsaws)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_6d = pd.Series(close_1d).ewm(span=6, min_periods=6).mean().values
    ema_6d_aligned = align_htf_to_ltf(prices, df_1d, ema_6d)
    
    # Volume filter: current volume > 2.0x 50-period average (high threshold = fewer trades)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or 
            np.isnan(ema_6d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND above 6-day EMA (uptrend) AND volume spike
            if close[i] > r1_6h[i] and close[i] > ema_6d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND below 6-day EMA (downtrend) AND volume spike
            elif close[i] < s1_6h[i] and close[i] < ema_6d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below S1 OR below 6-day EMA (trend change)
            if close[i] < s1_6h[i] or close[i] < ema_6d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above R1 OR above 6-day EMA (trend change)
            if close[i] > r1_6h[i] or close[i] > ema_6d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals