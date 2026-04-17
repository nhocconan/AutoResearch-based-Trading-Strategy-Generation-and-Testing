#!/usr/bin/env python3
"""
6h_RSI20_TrendFilter_V1
RSI(20) > 60 for long, RSI(20) < 40 for short with 6h trend filter from 1w EMA200.
Exit when RSI crosses back to neutral (40-60) or trend weakens.
Designed to capture momentum in trending markets while avoiding whipsaws in ranges.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # === RSI(20) ===
    delta = np.diff(close)
    delta = np.concatenate([[0], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=20, min_periods=20).mean().values
    avg_loss = pd.Series(loss).rolling(window=20, min_periods=20).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1w EMA200 for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or 
            np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: RSI > 60, price above 1w EMA200
            if (rsi[i] > 60 and 
                close[i] > ema_200_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: RSI < 40, price below 1w EMA200
            elif (rsi[i] < 40 and 
                  close[i] < ema_200_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: RSI < 40 OR price below 1w EMA200
            if (rsi[i] < 40 or 
                close[i] < ema_200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI > 60 OR price above 1w EMA200
            if (rsi[i] > 60 or 
                close[i] > ema_200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_RSI20_TrendFilter_V1"
timeframe = "6h"
leverage = 1.0