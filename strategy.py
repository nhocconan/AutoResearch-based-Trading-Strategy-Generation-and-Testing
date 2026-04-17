#!/usr/bin/env python3
"""
4h_GoldenCross_Reverse_v1
Golden Cross (EMA20 > EMA50) + RSI(14) < 30 for long, Death Cross + RSI > 70 for short.
Uses 1d timeframe for regime filter: only trade when price > 1d EMA200 (bull regime) for longs,
price < 1d EMA200 (bear regime) for shorts.
Exit when RSI crosses back to 50 or EMA cross reverses.
Designed to catch mean-reversion within the trend, avoiding counter-trend trades.
Target: 20-60 total trades over 4 years (5-15/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === EMA20 and EMA50 ===
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1d EMA200 for regime filter ===
    df_1d = get_htf_data(prices, '1d')
    ema200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20[i]) or 
            np.isnan(ema50[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(ema200_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Golden Cross + Oversold in bull regime: EMA20 > EMA50, RSI < 30, price > 1d EMA200
            if (ema20[i] > ema50[i] and 
                rsi[i] < 30 and 
                close[i] > ema200_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Death Cross + Overbought in bear regime: EMA20 < EMA50, RSI > 70, price < 1d EMA200
            elif (ema20[i] < ema50[i] and 
                  rsi[i] > 70 and 
                  close[i] < ema200_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: EMA cross reverses OR RSI > 50
            if (ema20[i] < ema50[i] or 
                rsi[i] > 50):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: EMA cross reverses OR RSI < 50
            if (ema20[i] > ema50[i] or 
                rsi[i] < 50):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_GoldenCross_Reverse_v1"
timeframe = "4h"
leverage = 1.0