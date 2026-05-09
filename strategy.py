#!/usr/bin/env python3
"""
6h_KAMA_Stochastic_Regime
- Primary: 6h KAMA crossover (10/30) for momentum
- HTF: 1w RSI(14) filter (bullish > 50, bearish < 50) to align with weekly trend
- Entry: KAMA fast > slow AND weekly RSI > 50 → long (0.25)
         KAMA fast < slow AND weekly RSI < 50 → short (-0.25)
- Exit: KAMA cross reverses
- Volume filter: 6h volume > 1.5x 20-period average to avoid low-vol whipsaws
- Designed for 6h timeframe with weekly trend filter to reduce false signals in chop
"""

name = "6h_KAMA_Stochastic_Regime"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h KAMA (10, 30) for momentum
    def kama(close, fast_len, slow_len):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast_len+1) - 2/(slow_len+1)) + 2/(slow_len+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_fast = kama(close, 10, 30)
    kama_slow = kama(close, 30, 30)  # slow uses same length for smoothing
    
    # 1w RSI(14) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate weekly RSI
    delta = np.diff(df_1w['close'], prepend=df_1w['close'][0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align weekly RSI to 6h timeframe (waits for weekly close)
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # KAMA slow + vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: KAMA fast > slow AND weekly RSI > 50 (bullish) + volume filter
            if (kama_fast[i] > kama_slow[i] and 
                rsi_aligned[i] > 50 and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA fast < slow AND weekly RSI < 50 (bearish) + volume filter
            elif (kama_fast[i] < kama_slow[i] and 
                  rsi_aligned[i] < 50 and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA cross turns bearish
            if kama_fast[i] < kama_slow[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA cross turns bullish
            if kama_fast[i] > kama_slow[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals