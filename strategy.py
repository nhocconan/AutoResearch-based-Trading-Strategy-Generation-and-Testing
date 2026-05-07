#!/usr/bin/env python3
name = "6h_WeeklyPivot_Breakout_1dTrend_Volume"
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
    
    # Load weekly data ONCE before loop for pivot points and trend filter
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    pivot_w = (high_w + low_w + close_w) / 3
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    r2_w = pivot_w + (high_w - low_w)
    s2_w = pivot_w - (high_w - low_w)
    r3_w = high_w + 2 * (pivot_w - low_w)
    s3_w = low_w - 2 * (high_w - pivot_w)
    r4_w = r3_w + (high_w - low_w)
    s4_w = s3_w - (high_w - low_w)
    
    # Align weekly pivot levels to 6h
    pivot_w_aligned = align_htf_to_ltf(prices, df_w, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_w, s1_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_w, r2_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_w, s2_w)
    r3_w_aligned = align_htf_to_ltf(prices, df_w, r3_w)
    s3_w_aligned = align_htf_to_ltf(prices, df_w, s3_w)
    r4_w_aligned = align_htf_to_ltf(prices, df_w, r4_w)
    s4_w_aligned = align_htf_to_ltf(prices, df_w, s4_w)
    
    # 1d EMA(34) for trend filter (load ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.3 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA(34)
    
    for i in range(start_idx, n):
        if np.isnan(ema_1d_aligned[i]) or np.isnan(pivot_w_aligned[i]) or np.isnan(r4_w_aligned[i]) or np.isnan(s4_w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above weekly R4 with uptrend and volume
            if (close[i] > r4_w_aligned[i] and close[i] > ema_1d_aligned[i] and vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below weekly S4 with downtrend and volume
            elif (close[i] < s4_w_aligned[i] and close[i] < ema_1d_aligned[i] and vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close back below weekly pivot or trend change
            if close[i] < pivot_w_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close back above weekly pivot or trend change
            if close[i] > pivot_w_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot points provide significant support/resistance levels. 
# Breaking above weekly R4 or below S4 with 1d trend alignment and volume confirmation 
# indicates strong momentum. Exit when price returns to weekly pivot or trend changes. 
# Weekly timeframe filters noise, suitable for 6h chart. Works in both bull (breakouts) 
# and bear (breakdowns) markets. Position size 0.25 manages risk. Target: 12-37 trades/year.