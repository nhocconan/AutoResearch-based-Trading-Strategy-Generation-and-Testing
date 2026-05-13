#!/usr/bin/env python3
"""
12h_KAMA_Trend_Volume_Confirmation
Hypothesis: KAMA adapts to market noise, providing a smooth trend line. Price above KAMA indicates uptrend, below indicates downtrend. Volume confirms the strength of the move. Weekly trend filter ensures alignment with higher timeframe momentum. Designed to work in both bull and bear markets by following the trend with volume confirmation, reducing whipsaws in choppy conditions. Target: 15-35 trades/year per symbol.
"""

name = "12h_KAMA_Trend_Volume_Confirmation"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA: Kaufman Adaptive Moving Average
    # Efficiency Ratio (ER) = |net change| / sum of absolute changes
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = np.abs(np.diff(close, prepend=close[0]))
    er = np.zeros_like(change)
    for i in range(1, len(change)):
        if np.sum(abs_change[i-9:i+1]) > 0:
            er[i] = np.abs(change[i] - change[i-10]) / np.sum(abs_change[i-9:i+1]) if i >= 10 else 0
        else:
            er[i] = 0
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume average (20-period)
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if np.sum(volume[max(0, i-19):i+1]) > 0 else 0
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    
    # Weekly trend filter: EMA50 on 1w data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1w = df_1w['close'].values > ema_50_1w
    downtrend_1w = df_1w['close'].values < ema_50_1w
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Volume confirmation: current volume > 1.5 * 20-period MA
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # LONG: price > KAMA, volume confirmation, weekly uptrend
            if close[i] > kama[i] and vol_confirm and uptrend_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price < KAMA, volume confirmation, weekly downtrend
            elif close[i] < kama[i] and vol_confirm and downtrend_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals