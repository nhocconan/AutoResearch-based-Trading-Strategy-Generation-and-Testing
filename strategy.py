#!/usr/bin/env python3
# 12h_KAMA_Direction_1dTrend_Volume
# Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) on 12h for trend direction, with 1d EMA34 as higher timeframe filter and volume confirmation. Enter long when KAMA turns up, price above KAMA, volume > 1.5x average, and 1d close > EMA34. Enter short when KAMA turns down, price below KAMA, volume > 1.5x average, and 1d close < EMA34. Exit when price crosses back over KAMA. Designed for 12h to target 12-37 trades/year. KAMA adapts to market noise, reducing whipsaws in ranging markets while capturing trends.

name = "12h_KAMA_Direction_1dTrend_Volume"
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
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate KAMA (12h)
    er_len = 10
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    
    change = np.abs(np.diff(close, n=er_len))
    vol = np.sum(np.abs(np.diff(close)), axis=1) if len(close) > 1 else np.array([])
    if len(vol) == 0:
        er = np.zeros_like(change)
    else:
        er = np.where(vol != 0, change / vol, 0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama = np.full(n, np.nan)
    kama[er_len] = close[er_len]
    for i in range(er_len + 1, n):
        kama[i] = kama[i-1] + sc[i-er_len] * (close[i] - kama[i-1])
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (close_1d[i] * 2 / (34 + 1)) + (ema_34_1d[i-1] * (1 - 2 / (34 + 1)))
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, er_len + 1)
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA turning up, price above KAMA, volume confirmation, 1d uptrend
            if kama[i] > kama[i-1] and close[i] > kama[i] and volume[i] > 1.5 * vol_ma[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA turning down, price below KAMA, volume confirmation, 1d downtrend
            elif kama[i] < kama[i-1] and close[i] < kama[i] and volume[i] > 1.5 * vol_ma[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price crosses back below KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price crosses back above KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals