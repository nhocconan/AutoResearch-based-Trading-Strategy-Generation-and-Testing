#!/usr/bin/env python3
"""
1h_4h1d_Triple_Confirmation_v1
Hypothesis: Combine 4h trend (EMA21), 1d momentum (RSI14), and 1h volatility breakout (ATR-based) 
for high-probability entries. Uses 4h for trend direction, 1d for momentum filter, 
and 1h for precise entry timing. Designed to work in both bull and bear markets by 
requiring alignment across timeframes. Target: 15-35 trades/year per symbol.
"""

name = "1h_4h1d_Triple_Confirmation_v1"
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
    
    # === 4H Data for Trend Direction ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # === 1D Data for Momentum Filter (RSI) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate RSI(14) on daily
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 1H Data for Entry Timing (ATR-based breakout) ===
    # Calculate ATR(14) for volatility
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr1 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], tr1])
    
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Donchian channel breakout (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(50, lookback)
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema21_4h_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long conditions:
            # 1. 4h trend up (price > EMA21)
            # 2. 1d momentum bullish (RSI > 50)
            # 3. 1h breakout above Donchian high with volume expansion (using price action proxy)
            if (close[i] > ema21_4h_aligned[i] and 
                rsi_1d_aligned[i] > 50 and 
                close[i] > highest_high[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions:
            # 1. 4h trend down (price < EMA21)
            # 2. 1d momentum bearish (RSI < 50)
            # 3. 1h breakdown below Donchian low
            elif (close[i] < ema21_4h_aligned[i] and 
                  rsi_1d_aligned[i] < 50 and 
                  close[i] < lowest_low[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: trend breaks down or momentum fades
            if close[i] < ema21_4h_aligned[i] or rsi_1d_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20  # maintain position
        elif position == -1:
            # Short exit: trend breaks up or momentum fades
            if close[i] > ema21_4h_aligned[i] or rsi_1d_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20  # maintain position
    
    return signals