#!/usr/bin/env python3
"""
1h_Time_Range_Reversal
Hypothesis: Trade mean reversions during high-liquidity London/NY overlap (08-12 UTC) and NY alone (12-16 UTC). Uses 4h RSI(14) for direction and 1h Stochastic for entry timing. Low-frequency (~20 trades/year) to avoid fee drag, works in ranging markets (2025+).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 4h data for RSI direction filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate 4h RSI(14)
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_4h)
    avg_loss = np.zeros_like(close_4h)
    for i in range(len(close_4h)):
        if i < 14:
            if i > 0:
                avg_gain[i] = np.mean(gain[:i+1])
                avg_loss[i] = np.mean(loss[:i+1])
            else:
                avg_gain[i] = gain[0]
                avg_loss[i] = loss[0]
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_4h = 100 - (100 / (1 + rs))
    
    # Align 4h RSI to 1h
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Calculate 1h Stochastic(14,3,3) for entry
    lowest_low = np.zeros_like(low)
    highest_high = np.zeros_like(high)
    for i in range(n):
        start_idx = max(0, i - 13)
        lowest_low[i] = np.min(low[start_idx:i+1])
        highest_high[i] = np.max(high[start_idx:i+1])
    
    stoch_k = np.where((highest_high - lowest_low) != 0, 
                       (close - lowest_low) / (highest_high - lowest_low) * 100, 50)
    
    stoch_d = np.zeros_like(stoch_k)
    for i in range(n):
        start_idx = max(0, i - 2)
        stoch_d[i] = np.mean(stoch_k[start_idx:i+1])
    
    # Session filter: 08-16 UTC (London/NY overlap + NY)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours < 16)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup
    
    for i in range(start_idx, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_4h_aligned[i]) or np.isnan(stoch_k[i]) or np.isnan(stoch_d[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: 4h RSI < 40 (not strong downtrend) + Stochastic bullish crossover from oversold
            if rsi_4h_aligned[i] < 40 and stoch_k[i-1] < stoch_d[i-1] and stoch_k[i] > stoch_d[i] and stoch_k[i] < 30:
                signals[i] = 0.20
                position = 1
            # Short: 4h RSI > 60 (not strong uptrend) + Stochastic bearish crossover from overbought
            elif rsi_4h_aligned[i] > 60 and stoch_k[i-1] > stoch_d[i-1] and stoch_k[i] < stoch_d[i] and stoch_k[i] > 70:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: Stochastic bearish crossover or hold max 8 hours
            if stoch_k[i] < stoch_d[i] or (i % 8 == 0 and i > start_idx):  # time-based exit every 8 bars
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: Stochastic bullish crossover or hold max 8 hours
            if stoch_k[i] > stoch_d[i] or (i % 8 == 0 and i > start_idx):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Time_Range_Reversal"
timeframe = "1h"
leverage = 1.0