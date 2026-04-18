#!/usr/bin/env python3
"""
4h_KAMA_Trend_Change_With_Volume
Hypothesis: KAMA adapts to market efficiency, providing early trend change signals. 
Combining KAMA direction with volume confirmation and ATR-based stops creates a robust 
trend-following system that works in both bull and bear markets by avoiding whipsaws 
through adaptive smoothing. Target: 20-30 trades/year on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) - 10 period
    # ER (Efficiency Ratio) = |change| / sum(|changes|)
    change = np.abs(np.diff(close, prepend=close[0]))
    er_num = np.abs(np.diff(close, n=10, prepend=close[:10]))
    er_den = np.zeros(n)
    for i in range(10, n):
        er_den[i] = np.sum(change[i-9:i+1])
    er = np.zeros(n)
    mask = er_den != 0
    er[mask] = er_num[mask] / er_den[mask]
    # SC (Smoothing Constant) = [ER * (fastest - slowest) + slowest]^2
    fastest = 2 / (2 + 1)   # EMA(2)
    slowest = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fastest - slowest) + slowest) ** 2
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA direction: 1 if close > KAMA, -1 if close < KAMA
    kama_dir = np.where(close > kama, 1, -1)
    
    # Volume confirmation: current volume > 1.8 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.8)
    
    # ATR for stop loss calculation
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if np.isnan(atr[i-1]):
            atr[i] = np.mean(tr[i-13:i+1])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA turns up with volume confirmation
            if kama_dir[i] == 1 and kama_dir[i-1] == -1 and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA turns down with volume confirmation
            elif kama_dir[i] == -1 and kama_dir[i-1] == 1 and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA turns down OR stop loss hit
            if kama_dir[i] == -1 or close[i] < (high[i-1] - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA turns up OR stop loss hit
            if kama_dir[i] == 1 or close[i] > (low[i-1] + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Trend_Change_With_Volume"
timeframe = "4h"
leverage = 1.0