#!/usr/bin/env python3
name = "12h_KAMA_1dTrend_VolumeSpike"
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
    volume = prices['volume'].values
    
    # KAMA parameters
    fast = 2
    slow = 30
    
    # Calculate ER (Efficiency Ratio) and SSC (Smoothing Constant)
    change = np.abs(np.diff(close, n=10))  # 10-period change
    vol = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # 10-period volatility
    # Avoid division by zero
    er = np.where(vol != 0, change / vol, 0)
    # Smoothing constants
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: volume > 1.5 * 20-period average
    vol_ma = np.full_like(volume, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Need 1d EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if 1d trend data not ready
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(kama[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA up + price above KAMA + 1d uptrend + volume spike
            if (kama[i] > kama[i-1] and 
                close[i] > kama[i] and 
                close[i] > ema34_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: KAMA down + price below KAMA + 1d downtrend + volume spike
            elif (kama[i] < kama[i-1] and 
                  close[i] < kama[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when KAMA turns down or price crosses below KAMA
            if (kama[i] < kama[i-1] or close[i] < kama[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when KAMA turns up or price crosses above KAMA
            if (kama[i] > kama[i-1] or close[i] > kama[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals