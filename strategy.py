#!/usr/bin/env python3
"""
4h_12hCamarilla_R3S3_Breakout_1dTrend_Volume
Hypothesis: Trade breakouts at daily Camarilla R3/S3 levels on 4h timeframe with 1d trend filter and volume confirmation.
Daily Camarilla levels provide reliable support/resistance. Breakouts aligned with daily trend and volume
should capture meaningful moves while avoiding noise. Works in bull/bear markets by following daily trend.
"""

name = "4h_12hCamarilla_R3S3_Breakout_1dTrend_Volume"
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
    
    # === Daily Camarilla Levels (R3, S3) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ph_d = df_1d['high'].values
    pl_d = df_1d['low'].values
    pc_d = df_1d['close'].values
    
    # Daily Camarilla levels
    r3_d = pc_d + 1.1 * (ph_d - pl_d)  # R3 level
    s3_d = pc_d - 1.1 * (ph_d - pl_d)  # S3 level
    
    # Align to 4h timeframe
    r3_4h = align_htf_to_ltf(prices, df_1d, r3_d)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3_d)
    
    # === 12h Trend Filter (EMA50) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # === Volume Filter (1.5x 20-period EMA on 4h) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers daily and 12h calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or np.isnan(ema50_4h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price closes above R3 with uptrend and volume
            if (close[i] > r3_4h[i] and 
                close[i] > ema50_4h[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price closes below S3 with downtrend and volume
            elif (close[i] < s3_4h[i] and 
                  close[i] < ema50_4h[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below S3 (mean reversion to support)
            if close[i] < s3_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price closes above R3 (mean reversion to resistance)
            if close[i] > r3_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals