#!/usr/bin/env python3
"""
4h_Camarilla_Pullback_Trend
Hypothesis: 4h pullback to Camarilla R3/S3 levels in direction of 1d trend (EMA34) with volume confirmation.
Enters on retest of key levels rather than breakouts, reducing false signals. Uses 1d timeframe for trend and levels.
Designed for low trade frequency (~30-50/year) to minimize fee drag. Works in bull/bear by following higher timeframe trend.
"""

name = "4h_Camarilla_Pullback_Trend"
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
    
    # === 1d Data for Trend Filter and Camarilla Levels ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA34 for trend
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Previous 1d bar's OHLC for Camarilla calculation
    ph_1d = high_1d  # previous 1d high
    pl_1d = low_1d   # previous 1d low
    pc_1d = df_1d['close'].values  # previous 1d close
    
    # Camarilla levels: R3, S3
    camarilla_r3 = pc_1d + 1.1 * (ph_1d - pl_1d) / 2
    camarilla_s3 = pc_1d - 1.1 * (ph_1d - pl_1d) / 2
    
    # Align Camarilla levels to 4h
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === Volume Filter: 1.5x 20-period EMA on 4h ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers 1d EMA34)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price pulls back to S3 in uptrend with volume confirmation
            if (close[i] >= s3_aligned[i] * 0.999 and close[i] <= s3_aligned[i] * 1.001 and 
                close[i] > ema34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price pulls back to R3 in downtrend with volume confirmation
            elif (close[i] >= r3_aligned[i] * 0.999 and close[i] <= r3_aligned[i] * 1.001 and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below S3 (break of pullback level)
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price closes above R3 (break of pullback level)
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals