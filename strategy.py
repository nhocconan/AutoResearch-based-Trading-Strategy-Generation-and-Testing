#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_Extreme_Volume
Hypothesis: KAMA adapts to market noise, providing reliable trend direction.
Combine with RSI extremes (<30 or >70) and volume confirmation to capture
trend reversals after exhaustion. Low-frequency design avoids whipsaws in
choppy markets while capturing strong directional moves. Works in bull via
trend continuation and in bear via mean-reversion bounces from oversold/overbought.
"""

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
    volume = prices['volume'].values
    
    # KAMA trend filter (1h)
    df_1h = get_htf_data(prices, '1h')
    close_1h = df_1h['close'].values
    
    # Efficiency Ratio and KAMA calculation
    er = np.full(len(close_1h), np.nan)
    for i in range(10, len(close_1h)):
        change = abs(close_1h[i] - close_1h[i-10])
        volatility = np.sum(np.abs(np.diff(close_1h[i-10:i+1])))
        if volatility != 0:
            er[i] = change / volatility
        else:
            er[i] = 1.0
    
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    kama = np.full(len(close_1h), np.nan)
    if len(close_1h) > 0:
        kama[0] = close_1h[0]
        for i in range(1, len(close_1h)):
            kama[i] = kama[i-1] + sc[i] * (close_1h[i] - kama[i-1])
    
    kama_1h_aligned = align_htf_to_ltf(prices, df_1h, kama)
    
    # RSI (14) on 4h close
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[0:14])
            avg_loss[i] = np.mean(loss[0:14])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(kama_1h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA (uptrend) + RSI oversold + volume
            if (close[i] > kama_1h_aligned[i] and rsi[i] < 30 and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend) + RSI overbought + volume
            elif (close[i] < kama_1h_aligned[i] and rsi[i] > 70 and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI overbought or price breaks below KAMA
            if (rsi[i] > 70 or close[i] < kama_1h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI oversold or price breaks above KAMA
            if (rsi[i] < 30 or close[i] > kama_1h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Direction_RSI_Extreme_Volume"
timeframe = "4h"
leverage = 1.0