#!/usr/bin/env python3
name = "1d_WeeklyPivot_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prrices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    trend_up = close > ema50_1w_aligned
    trend_down = close < ema50_1w_aligned
    
    # Weekly pivot levels (R1/S1) - more commonly used
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].shift(1).values
    close_1d_prev = np.concatenate([[close_1d_prev[0]], close_1d_prev[:-1]])
    
    # Weekly pivot points (weekly high/low/close)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_prev = df_1w['close'].shift(1).values
    close_1w_prev = np.concatenate([[close_1w_prev[0]], close_1w_prev[:-1]])
    
    pivot = (high_1w + low_1w + close_1w_prev) / 3
    R1 = pivot + (high_1w - low_1w) * 1.1 / 4  # Weekly R1
    S1 = pivot - (high_1w - low_1w) * 1.1 / 4  # Weekly S1
    
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    
    # Volume confirmation: spike > 1.5x 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_spike = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close breaks above R1 with volume spike and 1w uptrend
            if close[i] > R1_aligned[i] and vol_spike[i] and trend_up[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S1 with volume spike and 1w downtrend
            elif close[i] < S1_aligned[i] and vol_spike[i] and trend_down[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below S1 or trend turns down
            if close[i] < S1_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above R1 or trend turns up
            if close[i] > R1_aligned[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot R1/S1 breakouts with 1-week trend filter and volume confirmation capture institutional moves.
# Long when price breaks above weekly R1 (strong resistance) with volume confirmation in 1-week uptrend.
# Short when price breaks below weekly S1 (strong support) with volume confirmation in 1-week downtrend.
# Weekly pivots are more robust than daily pivots for longer-term trends.
# Volume spike (>1.5x 50-day average) ensures conviction behind the breakout.
# Designed for 1d timeframe to target 7-25 trades/year, avoiding overtrading.
# Works in bull markets (breaks above R1 in uptrend) and bear markets (breaks below S1 in downtrend).