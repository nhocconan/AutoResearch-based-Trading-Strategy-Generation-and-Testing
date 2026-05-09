#!/usr/bin/env python3
name = "1D_Weekly_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "1d"
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
    
    # Get weekly data for Camarilla pivot levels and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 12:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point and Camarilla levels (R1, S1)
    pivot = (high_1w + low_1w + close_1w) / 3
    range_ = high_1w - low_1w
    r1 = pivot + (range_ * 1.1 / 2)
    s1 = pivot - (range_ * 1.1 / 2)
    
    # Weekly EMA12 for trend filter
    ema12_1w = pd.Series(close_1w).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # Align to daily
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    ema12_1w_aligned = align_htf_to_ltf(prices, df_1w, ema12_1w)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema12_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R1 + above weekly EMA12 + volume confirmation
            if close[i] > r1_aligned[i] and close[i] > ema12_1w_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 + below weekly EMA12 + volume confirmation
            elif close[i] < s1_aligned[i] and close[i] < ema12_1w_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below weekly EMA12 (trend change)
            if close[i] < ema12_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above weekly EMA12 (trend change)
            if close[i] > ema12_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals