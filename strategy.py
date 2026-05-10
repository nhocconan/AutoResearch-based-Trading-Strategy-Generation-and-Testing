#!/usr/bin/env python3
# 1d_KAMA_Direction_WeeklyTrend_Volume
# Hypothesis: Use 1d KAMA direction filtered by 1-week EMA trend and volume spike.
# Long when KAMA turns up, price > weekly EMA50, and volume > 1.5x average.
# Short when KAMA turns down, price < weekly EMA50, and volume > 1.5x average.
# Exit when KAMA reverses direction.
# Designed for 7-25 trades/year on daily timeframe to avoid fee drag.
# Works in bull/bear via weekly trend filter and volatility-adjusted entry.

name = "1d_KAMA_Direction_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA(10,2,30) - ER=10, fast=2, slow=30
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.zeros(n)
    for i in range(1, n):
        if volatility[i-1] > 0:
            er[i] = change[i] / volatility[i-1]
        else:
            er[i] = 0
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA direction: 1 if rising, -1 if falling
    kama_dir = np.zeros(n)
    for i in range(1, n):
        if kama[i] > kama[i-1]:
            kama_dir[i] = 1
        elif kama[i] < kama[i-1]:
            kama_dir[i] = -1
        else:
            kama_dir[i] = kama_dir[i-1]
    
    # Get weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup for weekly EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA turning up, price above weekly EMA50, volume spike
            if kama_dir[i] == 1 and kama_dir[i-1] == -1:  # KAMA just turned up
                if close[i] > ema_50_1w_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = 0.25
                    position = 1
            # Short: KAMA turning down, price below weekly EMA50, volume spike
            elif kama_dir[i] == -1 and kama_dir[i-1] == 1:  # KAMA just turned down
                if close[i] < ema_50_1w_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: KAMA turns down
            if kama_dir[i] == -1 and kama_dir[i-1] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: KAMA turns up
            if kama_dir[i] == 1 and kama_dir[i-1] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals