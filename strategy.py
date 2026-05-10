#!/usr/bin/env python3
# 1d_KAMA_Direction_WeeklyTrend_Volume
# Hypothesis: Go long when KAMA direction is up, price above weekly EMA34, and volume > 1.5x average.
# Go short when KAMA direction is down, price below weekly EMA34, and volume > 1.5x average.
# Exit when KAMA direction changes or weekly trend fails.
# Uses weekly trend filter to avoid counter-trend trades. Designed for 1d timeframe to target 7-25 trades/year.
# KAMA adapts to market noise, reducing whipsaws in sideways markets.

name = "1d_KAMA_Direction_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Calculate 1w EMA34 for trend filter (using HTF data)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = np.full(len(close_1w), np.nan)
    for i in range(34, len(close_1w)):
        ema_34_1w[i] = np.mean(close_1w[i-34:i])  # Simple MA for efficiency
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate KAMA (2, 10, 30) - ER=2, fast=2, slow=30
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.zeros(n)
    er[1:] = change[1:] / (np.sum(np.abs(np.diff(close, prepend=close[0]))[np.arange(1, n) - 9:np.arange(1, n)+1], axis=1) + 1e-10)
    # Simplified ER calculation for efficiency
    er_smooth = np.zeros(n)
    for i in range(10, n):
        er_smooth[i] = np.mean(er[i-9:i+1])
    sc = (er_smooth * (2/2 - 2/30) + 2/30) ** 2
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_dir = np.sign(np.diff(kama, prepend=0))
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure sufficient warmup for KAMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(kama_dir[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA up, price above weekly EMA34, volume confirmation
            if kama_dir[i] > 0 and close[i] > ema_34_1w_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, price below weekly EMA34, volume confirmation
            elif kama_dir[i] < 0 and close[i] < ema_34_1w_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: KAMA down or price below weekly EMA34
            if kama_dir[i] < 0 or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: KAMA up or price above weekly EMA34
            if kama_dir[i] > 0 or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals