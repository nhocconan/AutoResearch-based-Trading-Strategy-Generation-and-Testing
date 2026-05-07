#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume_Spike_v2"
timeframe = "4h"
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
    
    # 1d trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    trend_up = close > ema50_1d_aligned
    trend_down = close < ema50_1d_aligned
    
    # Daily Camarilla pivot levels (R3/S3)
    df_1d_full = get_htf_data(prices, '1d')
    if len(df_1d_full) < 2:
        return np.zeros(n)
    
    high_1d = df_1d_full['high'].values
    low_1d = df_1d_full['low'].values
    close_1d_prev = df_1d_full['close'].shift(1).values
    close_1d_prev = np.concatenate([[close_1d_prev[0]], close_1d_prev[:-1]])
    
    R3 = close_1d_prev + (high_1d - low_1d) * 1.1 / 4
    S3 = close_1d_prev - (high_1d - low_1d) * 1.1 / 4
    R3_aligned = align_htf_to_ltf(prices, df_1d_full, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d_full, S3)
    
    # Volume confirmation: spike > 2.5x 20-period average (more stringent)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close breaks above R3 with volume spike and 1d uptrend
            if close[i] > R3_aligned[i] and vol_spike[i] and trend_up[i]:
                signals[i] = 0.30
                position = 1
            # Short: Close breaks below S3 with volume spike and 1d downtrend
            elif close[i] < S3_aligned[i] and vol_spike[i] and trend_down[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: Close below S3 or trend turns down
            if close[i] < S3_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: Close above R3 or trend turns up
            if close[i] > R3_aligned[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: Camarilla R3/S3 breakouts with 1d trend filter and volume spike (2.5x) capture strong institutional moves.
# Long when price breaks above R3 (strong resistance) with volume confirmation in 1d uptrend.
# Short when price breaks below S3 (strong support) with volume confirmation in 1d downtrend.
# R3/S3 are stronger levels than R1/S1, leading to fewer but higher-quality trades.
# Volume spike (>2.5x average) ensures conviction behind the breakout.
# Designed for 4h timeframe to target 20-30 trades/year, avoiding overtrading.
# Works in bull markets (breaks above R3 in uptrend) and bear markets (breaks below S3 in downtrend).