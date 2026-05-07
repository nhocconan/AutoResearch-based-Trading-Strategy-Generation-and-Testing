#!/usr/bin/env python3
name = "6h_Weekly_Pivot_Breakout_1dTrend_Volume"
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
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Weekly high, low, close from previous week
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    # Weekly pivot point and resistance/support levels
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    R1 = weekly_pivot + (prev_week_high - prev_week_low)
    S1 = weekly_pivot - (prev_week_high - prev_week_low)
    R2 = weekly_pivot + 2 * (prev_week_high - prev_week_low)
    S2 = weekly_pivot - 2 * (prev_week_high - prev_week_low)
    R3 = weekly_pivot + 3 * (prev_week_high - prev_week_low)
    S3 = weekly_pivot - 3 * (prev_week_high - prev_week_low)
    
    # Align to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1w, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1w, S2)
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA20 for trend filter
    ema20_1d = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Volume spike: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 5)  # Wait for volume MA and weekly data
    
    for i in range(start_idx, n):
        if np.isnan(pivot_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(ema20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close breaks above R1 with volume spike in 1d uptrend
            if close[i] > R1_aligned[i] and vol_spike[i] and close[i] > ema20_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S1 with volume spike in 1d downtrend
            elif close[i] < S1_aligned[i] and vol_spike[i] and close[i] < ema20_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below S1 or trend turns down
            if close[i] < S1_aligned[i] or close[i] < ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above R1 or trend turns up
            if close[i] > R1_aligned[i] or close[i] > ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot point breakout on 6h with 1d EMA20 trend filter and volume confirmation.
# Long when price breaks above weekly R1 (bullish breakout) with volume spike in 1d uptrend.
# Short when price breaks below weekly S1 (bearish breakdown) with volume spike in 1d downtrend.
# Uses weekly pivot levels which are more significant than daily pivots for capturing major trend changes.
# Weekly pivots act as strong support/resistance levels that institutions watch.
# Volume spike (>2.0x average) ensures conviction behind the move.
# Designed for 6h timeframe to target 50-150 total trades over 4 years, avoiding overtrading.
# Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend).