#!/usr/bin/env python3
name = "1h_PivotBreakout_4hTrend_1dVolumeFilter"
timeframe = "1h"
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
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h EMA(20) for trend filter
    ema_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Load 1d data ONCE before loop for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Pivot Points from previous 1d
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    
    # Align Pivot levels to 1h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume filter: 1h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ma
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for EMA and Pivots
    
    for i in range(start_idx, n):
        if np.isnan(ema_4h_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R1 with 4h uptrend and volume, during session
            if (close[i] > r1_aligned[i] and close[i] > ema_4h_aligned[i] and vol_filter[i] and session_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: Break below S1 with 4h downtrend and volume, during session
            elif (close[i] < s1_aligned[i] and close[i] < ema_4h_aligned[i] and vol_filter[i] and session_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: Close below S1 or trend change
            if close[i] < s1_aligned[i] or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: Close above R1 or trend change
            if close[i] > r1_aligned[i] or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h Pivot Breakout with 4h EMA(20) trend filter and 1d volume confirmation.
# Uses daily pivot points (R1/S1) as key support/resistance levels.
# 4h EMA ensures alignment with higher timeframe trend, reducing whipsaw.
# Volume filter confirms institutional participation.
# Session filter (08-20 UTC) reduces noise trades.
# Position size 0.20 limits drawdown.
# Target: 15-30 trades/year per symbol (60-120 total over 4 years).