#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_Breakout_1dTrend_Volume_Confirmation
Hypothesis: Combine daily Camarilla pivot levels with 1d trend filter and volume confirmation for breakout trades. Works in bull/bear by aligning with higher timeframe trend. Uses discrete position sizing to minimize fee churn. Target: 20-40 trades/year on 4h.
"""

name = "4h_Camarilla_Pivot_Breakout_1dTrend_Volume_Confirmation"
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
    
    # === Daily OHLC for Camarilla Pivots ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # H, L, C from previous completed daily candle
    ph = df_1d['high'].values
    pl = df_1d['low'].values
    pc = df_1d['close'].values
    
    # Camarilla levels: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We use R3, R2, S3, S2 for breakout/retest
    camarilla_r3 = pc + (ph - pl) * 1.1 / 4
    camarilla_r2 = pc + (ph - pl) * 1.1 / 6
    camarilla_s2 = pc - (ph - pl) * 1.1 / 6
    camarilla_s3 = pc - (ph - pl) * 1.1 / 4
    
    # Align to 4h timeframe (these levels are valid until next daily candle)
    r3_4h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    r2_4h = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    s2_4h = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    s3_4h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === Daily Trend Filter (EMA34) ===
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Volume Filter (1.5x 20-period EMA on 4h) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers daily calculations)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_4h[i]) or np.isnan(r2_4h[i]) or np.isnan(s2_4h[i]) or np.isnan(s3_4h[i]) or
            np.isnan(ema34_4h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R3 with uptrend and volume
            if (close[i] > r3_4h[i] and 
                close[i] > ema34_4h[i] and 
                volume_ok[i]):
                signals[i] = 0.30
                position = 1
            # Short breakdown: price breaks below S3 with downtrend and volume
            elif (close[i] < s3_4h[i] and 
                  close[i] < ema34_4h[i] and 
                  volume_ok[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price breaks below R2 (profit target or reversal)
            if close[i] < r2_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30  # maintain position
        elif position == -1:
            # Short exit: price breaks above S2 (profit target or reversal)
            if close[i] > s2_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30  # maintain position
    
    return signals