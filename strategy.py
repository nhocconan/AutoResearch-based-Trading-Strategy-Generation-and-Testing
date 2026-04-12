#!/usr/bin/env python3
"""
4h_1d_RSI_Reversion_Pivot_Touch_v1
Hypothesis: In both bull and bear markets, price often reverts to daily pivot after touching R3/S3 levels when RSI is extreme. Uses RSI(14) for mean reversion signals with daily pivot as target, limited to 1-2 trades per week via strict entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_RSI_Reversion_Pivot_Touch_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot calculation
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels (daily)
    r3_1d = close_1d + range_1d * 1.1
    s3_1d = close_1d - range_1d * 1.1
    
    # Align daily levels to 4h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # RSI(14) calculation
    if len(close) >= 14:
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros(n)
        avg_loss = np.zeros(n)
        avg_gain[13] = np.mean(gain[1:14])
        avg_loss[13] = np.mean(loss[1:14])
        
        for i in range(14, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        rsi[:13] = 50  # neutral before enough data
    else:
        rsi = np.full(n, 50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(pivot_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Mean reversion signals at extreme RSI touching S3/R3
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Price touching S3 (long setup) or R3 (short setup)
        price_at_s3 = low[i] <= s3_1d_aligned[i] * 1.001  # allow small tolerance
        price_at_r3 = high[i] >= r3_1d_aligned[i] * 0.999
        
        long_setup = rsi_oversold and price_at_s3
        short_setup = rsi_overbought and price_at_r3
        
        # Exit when price returns to daily pivot
        exit_long = close[i] >= pivot_1d_aligned[i] * 0.999
        exit_short = close[i] <= pivot_1d_aligned[i] * 1.001
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals