#!/usr/bin/env python3
name = "12h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    trend_up = close > ema20_1d_aligned
    trend_down = close < ema20_1d_aligned
    
    # 12h Camarilla pivot levels (R1/S1)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_prev = df_12h['close'].shift(1).values
    close_12h_prev = np.concatenate([[close_12h_prev[0]], close_12h_prev[:-1]])
    
    R1 = close_12h_prev + (high_12h - low_12h) * 1.1 / 12
    S1 = close_12h_prev - (high_12h - low_12h) * 1.1 / 12
    R1_aligned = align_htf_to_ltf(prices, df_12h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_12h, S1)
    
    # Volume confirmation: spike > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema20_1d_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close breaks above R1 with volume spike and 1d uptrend
            if close[i] > R1_aligned[i] and vol_spike[i] and trend_up[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S1 with volume spike and 1d downtrend
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

# Hypothesis: Camarilla R1/S1 breakouts on 12h with 1d trend filter and volume capture institutional moves.
# Long when price breaks above R1 (minor resistance) with volume confirmation in 1d uptrend.
# Short when price breaks below S1 (minor support) with volume confirmation in 1d downtrend.
# R1/S1 levels provide more frequent but still high-quality breaks compared to R3/S3.
# Volume spike (>1.5x average) ensures conviction behind the breakout.
# Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year).
# Works in bull markets (breaks above R1 in uptrend) and bear markets (breaks below S1 in downtrend).