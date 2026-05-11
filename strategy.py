#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_Volume
Hypothesis: Trade breakouts at Camarilla R1/S1 levels on 4h timeframe with 12h trend filter and volume confirmation. 
Camarilla levels provide precise support/resistance. Breakouts in direction of 12h trend with volume confirmation 
should capture momentum moves while avoiding false breakouts. Works in bull/bear markets by aligning with 12h trend.
Target: 20-50 trades/year on 4h timeframe to minimize fee drag.
"""

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume"
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
    
    # === Daily OHLC for Camarilla Pivot Points ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ph_d = df_1d['high'].values
    pl_d = df_1d['low'].values
    pc_d = df_1d['close'].values
    
    # Camarilla Levels (based on previous day's OHLC)
    # R1 = Close + 1.1 * (High - Low) / 12
    # S1 = Close - 1.1 * (High - Low) / 12
    r1_d = pc_d + 1.1 * (ph_d - pl_d) / 12
    s1_d = pc_d - 1.1 * (ph_d - pl_d) / 12
    
    # Align to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1_d)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1_d)
    
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
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or np.isnan(ema50_4h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price closes above R1 with uptrend and volume
            if (close[i] > r1_4h[i] and 
                close[i] > ema50_4h[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price closes below S1 with downtrend and volume
            elif (close[i] < s1_4h[i] and 
                  close[i] < ema50_4h[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below S1 (mean reversion to support)
            if close[i] < s1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price closes above R1 (mean reversion to resistance)
            if close[i] > r1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals