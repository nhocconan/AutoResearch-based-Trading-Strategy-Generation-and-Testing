#!/usr/bin/env python3
"""
12h_Keltner_RSI_Combo_v1
Hypothesis: Combines Keltner Channel breakout with RSI momentum to capture trend continuation in both bull and bear markets. 
Uses 1w trend filter to align with higher timeframe momentum. Designed for low trade frequency (~15-30/year) to minimize fee drag.
"""

name = "12h_Keltner_RSI_Combo_v1"
timeframe = "12h"
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
    
    # === 1W Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 50-period EMA on weekly for trend
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === Keltner Channel (20, 2.0) on 12h ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # EMA20 for middle line
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    # ATR for bands
    tr1 = np.maximum(high_12h[1:], low_12h[:-1]) - np.minimum(low_12h[1:], high_12h[:-1])
    tr1 = np.concatenate([[np.nan], tr1])
    atr = pd.Series(tr1).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    upper_keltner = ema20_12h + 2.0 * atr
    lower_keltner = ema20_12h - 2.0 * atr
    
    upper_keltner_aligned = align_htf_to_ltf(prices, df_12h, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_12h, lower_keltner)
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # === RSI(14) on 12h ===
    delta = np.diff(close_12h, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_12h, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(upper_keltner_aligned[i]) or 
            np.isnan(lower_keltner_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Keltner + RSI > 50 + weekly uptrend
            if (close[i] > upper_keltner_aligned[i] and 
                rsi_aligned[i] > 50 and 
                ema50_1w_aligned[i] < close[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Keltner + RSI < 50 + weekly downtrend
            elif (close[i] < lower_keltner_aligned[i] and 
                  rsi_aligned[i] < 50 and 
                  ema50_1w_aligned[i] > close[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below middle Keltner OR RSI < 40
            if (close[i] < ema20_12h_aligned[i] or 
                rsi_aligned[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price breaks above middle Keltner OR RSI > 60
            if (close[i] > ema20_12h_aligned[i] or 
                rsi_aligned[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals