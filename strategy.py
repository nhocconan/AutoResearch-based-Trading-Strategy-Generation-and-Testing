#!/usr/bin/env python3
"""
1D_WeeklyPivot_Breakout_1wTrend_Volume
Hypothesis: Breakouts at weekly pivot points (R1/S1) with volume confirmation and weekly trend alignment capture long-term trends. Weekly filters ensure low trade frequency (<10/year) to minimize fee decay while capturing major moves in both bull and bear markets. Uses volume surge (>2x 20-day avg) to confirm institutional participation.
"""

name = "1D_WeeklyPivot_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous weekly bar for pivot calculation (using weekly close)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot points (standard formula)
    pivot = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    s1 = pivot - range_1w  # S1 = P - (H - L)
    r1 = pivot + range_1w  # R1 = P + (H - L)
    
    # Align to daily timeframe (wait for weekly bar to close)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    
    # Weekly trend: EMA 21 on weekly close
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Volume filter: volume > 2x 20-day average (strict to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 20)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(ema_21_1w_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        is_uptrend = close[i] > ema_21_1w_aligned[i]
        is_downtrend = close[i] < ema_21_1w_aligned[i]
        
        if position == 0:
            # Long entry: Price breaks above R1 + volume confirmation + weekly uptrend
            if (close[i] > r1_aligned[i] and 
                volume[i] > vol_threshold[i] and 
                is_uptrend):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below S1 + volume confirmation + weekly downtrend
            elif (close[i] < s1_aligned[i] and 
                  volume[i] > vol_threshold[i] and 
                  is_downtrend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price crosses below S1 (opposite side)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price crosses above R1 (opposite side)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals