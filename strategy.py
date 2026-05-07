#!/usr/bin/env python3
name = "1d_WeeklyPivot_Breakout_1wTrend_Volume"
timeframe = "1d"
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
    
    # 1w trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    trend_up = close > ema20_1w_aligned
    trend_down = close < ema20_1w_aligned
    
    # Weekly pivot levels (from previous week)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_prev = df_1w['close'].shift(1).values
    close_1w_prev = np.concatenate([[close_1w_prev[0]], close_1w_prev[:-1]])
    
    # Weekly Pivot: (H + L + C) / 3
    pivot = (high_1w + low_1w + close_1w_prev) / 3
    # Weekly R1: 2*P - L
    R1 = 2 * pivot - low_1w
    # Weekly S1: 2*P - H
    S1 = 2 * pivot - high_1w
    
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    
    # Volume confirmation: spike > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema20_1w_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]):
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

# Hypothesis: Weekly pivot breakouts with weekly trend filter and volume spike capture strong institutional moves.
# Long when price breaks above weekly R1 (strong resistance) with volume confirmation in weekly uptrend.
# Short when price breaks below weekly S1 (strong support) with volume confirmation in weekly downtrend.
# Weekly pivots are significant levels that institutions watch, leading to fewer but higher-quality trades.
# Volume spike (>2x average) ensures conviction behind the breakout.
# Designed for 1d timeframe to target 20-40 trades/year, avoiding overtrading.
# Works in bull markets (breaks above R1 in uptrend) and bear markets (breaks below S1 in downtrend).