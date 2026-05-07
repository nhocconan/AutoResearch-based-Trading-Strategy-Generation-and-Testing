#!/usr/bin/env python3
name = "1h_Camarilla_R3S3_Breakout_4hTrend_Volume"
timeframe = "1h"
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
    
    # 4h trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    trend_up = close > ema50_4h_aligned
    trend_down = close < ema50_4h_aligned
    
    # Daily Camarilla pivot levels (R3/S3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].shift(1).values
    close_1d_prev = np.concatenate([[close_1d_prev[0]], close_1d_prev[:-1]])
    
    R3 = close_1d_prev + (high_1d - low_1d) * 1.1 / 4
    S3 = close_1d_prev - (high_1d - low_1d) * 1.1 / 4
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation: spike > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_ok = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema50_4h_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if session_ok[i]:
            if position == 0:
                # Long: Close breaks above R3 with volume spike and 4h uptrend
                if close[i] > R3_aligned[i] and vol_spike[i] and trend_up[i]:
                    signals[i] = 0.20
                    position = 1
                # Short: Close breaks below S3 with volume spike and 4h downtrend
                elif close[i] < S3_aligned[i] and vol_spike[i] and trend_down[i]:
                    signals[i] = -0.20
                    position = -1
            elif position == 1:
                # Exit: Close below S3 or trend turns down
                if close[i] < S3_aligned[i] or not trend_up[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit: Close above R3 or trend turns up
                if close[i] > R3_aligned[i] or not trend_down[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
        else:
            # Outside session: flatten if in position
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals

# Hypothesis: Camarilla R3/S3 breakouts with 4h trend filter and volume spike capture strong institutional moves.
# Long when price breaks above R3 (strong resistance) with volume confirmation in 4h uptrend.
# Short when price breaks below S3 (strong support) with volume confirmation in 4h downtrend.
# R3/S3 are stronger levels than R1/S1, leading to fewer but higher-quality trades.
# Volume spike (>2x average) ensures conviction behind the breakout.
# Session filter (08-20 UTC) reduces noise trades during low-volume periods.
# Designed for 1h timeframe to target 15-37 trades/year, avoiding excessive frequency.
# Works in bull markets (breaks above R3 in uptrend) and bear markets (breaks below S3 in downtrend).