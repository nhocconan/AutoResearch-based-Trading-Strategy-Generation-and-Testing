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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Weekly Pivot Points from previous week
    prev_high_w = df_1w['high'].shift(1).values
    prev_low_w = df_1w['low'].shift(1).values
    prev_close_w = df_1w['close'].shift(1).values
    
    pivot_w = (prev_high_w + prev_low_w + prev_close_w) / 3
    r4_w = pivot_w + 3 * (prev_high_w - prev_low_w)  # R4 = pivot + 3*(H-L)
    s4_w = pivot_w - 3 * (prev_high_w - prev_low_w)  # S4 = pivot - 3*(H-L)
    
    # Align Weekly Pivot levels to 6h
    pivot_w_aligned = align_htf_to_ltf(prices, df_1w, pivot_w)
    r4_w_aligned = align_htf_to_ltf(prices, df_1w, r4_w)
    s4_w_aligned = align_htf_to_ltf(prices, df_1w, s4_w)
    
    # Load daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA(20) for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for EMA and Vol MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_1d_aligned[i]) or np.isnan(r4_w_aligned[i]) or np.isnan(s4_w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above weekly R4 with 1d uptrend and volume
            if (close[i] > r4_w_aligned[i] and close[i] > ema_1d_aligned[i] and vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below weekly S4 with 1d downtrend and volume
            elif (close[i] < s4_w_aligned[i] and close[i] < ema_1d_aligned[i] and vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below weekly pivot or trend change
            if close[i] < pivot_w_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above weekly pivot or trend change
            if close[i] > pivot_w_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly Pivot R4/S4 breakout with 1d EMA trend filter and volume confirmation.
# Weekly R4/S4 represent strong monthly support/resistance. Breaking these levels indicates
# significant momentum. 1d EMA(20) ensures alignment with daily trend, reducing whipsaw.
# Volume filter confirms institutional participation. Position size 0.25 limits drawdown.
# Target: 10-30 trades/year on 6h timeframe. Works in both bull (breakouts) and bear (breakdowns).