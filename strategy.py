#!/usr/bin/env python3
"""
4h_KAMA_RSI_Trend_v1
KAMA direction (10/2) + RSI(14) filter (50/50) + 1d EMA50 trend filter.
Long: KAMA rising + RSI>50 + price>1d EMA50.
Short: KAMA falling + RSI<50 + price<1d EMA50.
Exit when KAMA direction reverses.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # === KAMA(10,2) ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period change
    abs_change = np.abs(np.diff(close, n=1))  # 1-period change
    er_num = np.concatenate([[np.nan]*9, change])  # align to same length
    er_den = np.concatenate([[np.nan]*9, np.cumsum(abs_change)[9:]])  # sum of abs changes
    er = er_num / (er_den + 1e-10)
    er = np.where(er_den == 0, 0, er)  # avoid division by zero
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # KAMA direction (1: rising, -1: falling, 0: flat)
    kama_dir = np.diff(kama, prepend=kama[0])
    kama_dir = np.where(kama_dir > 0, 1, np.where(kama_dir < 0, -1, 0))
    
    # === RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1d EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_dir[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: KAMA rising, RSI>50, price above 1d EMA50
            if (kama_dir[i] == 1 and 
                rsi[i] > 50 and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: KAMA falling, RSI<50, price below 1d EMA50
            elif (kama_dir[i] == -1 and 
                  rsi[i] < 50 and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: KAMA falling
            if kama_dir[i] == -1:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA rising
            if kama_dir[i] == 1:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_RSI_Trend_v1"
timeframe = "4h"
leverage = 1.0