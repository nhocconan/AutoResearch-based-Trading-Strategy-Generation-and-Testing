#!/usr/bin/env python3
"""
4h_1d_RSI_2DMA_Cross_v1
Long when 1d EMA21 > EMA50 and RSI14 > 50; short when EMA21 < EMA50 and RSI14 < 50.
Exit when RSI crosses back to neutral (40-60).
Uses 1d trend filter to align with higher timeframe momentum.
Target: 20-40 trades per year (~80-160 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1d EMA21 and EMA50 ===
    df_1d = get_htf_data(prices, '1d')
    ema21_1d = pd.Series(df_1d['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or 
            np.isnan(ema21_1d_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: EMA21 > EMA50 and RSI > 50
            if (ema21_1d_aligned[i] > ema50_1d_aligned[i] and 
                rsi[i] > 50):
                signals[i] = 0.25
                position = 1
                continue
            # Short: EMA21 < EMA50 and RSI < 50
            elif (ema21_1d_aligned[i] < ema50_1d_aligned[i] and 
                  rsi[i] < 50):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: RSI < 40 (overbought exit)
            if rsi[i] < 40:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI > 60 (oversold exit)
            if rsi[i] > 60:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_RSI_2DMA_Cross_v1"
timeframe = "4h"
leverage = 1.0