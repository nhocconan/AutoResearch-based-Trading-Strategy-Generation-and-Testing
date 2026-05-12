#!/usr/bin/env python3
name = "6h_WeeklyPivot_Reversion_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY PIVOT DATA ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    # Support 1 = 2*P - H, Resistance 1 = 2*P - L
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    r1_1w = 2 * pivot_1w - high_1w
    s1_1w = 2 * pivot_1w - low_1w
    
    # Align weekly pivot levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # === DAILY TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 50-period EMA for daily trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or 
            np.isnan(ema50_1d_6h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price near weekly S1 support + above daily trend + volume confirmation
            if (close[i] <= s1_6h[i] * 1.02 and  # Within 2% of S1
                close[i] > ema50_1d_6h[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price near weekly R1 resistance + below daily trend + volume confirmation
            elif (close[i] >= r1_6h[i] * 0.98 and  # Within 2% of R1
                  close[i] < ema50_1d_6h[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price reaches weekly pivot OR below daily trend
            if close[i] >= pivot_6h[i] or close[i] < ema50_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches weekly pivot OR above daily trend
            if close[i] <= pivot_6h[i] or close[i] > ema50_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals