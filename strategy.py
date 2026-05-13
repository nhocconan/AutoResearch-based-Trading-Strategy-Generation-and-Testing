#!/usr/bin/env python3
"""
4h_KAMA_Direction_Volume_Confirmation
Hypothesis: Kaufman Adaptive Moving Average (KAMA) adapts to market volatility, providing a trend-following signal that reduces whipsaw in choppy markets. 
Long when price > KAMA and volume > 1.5x average volume; short when price < KAMA and volume > 1.5x average volume.
Uses 12h EMA50 trend filter to align with higher timeframe direction. Volume confirmation ensures momentum behind moves.
Designed to work in both bull and bear markets by following the trend with adaptive smoothing.
Target: 20-40 trades/year per symbol.
"""

name = "4h_KAMA_Direction_Volume_Confirmation"
timeframe = "4h"
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
    
    # KAMA calculation
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder, will compute properly below
    # Recalculate volatility as rolling sum of absolute changes
    volatility = pd.Series(change).rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 12h trend filter: EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_12h = df_12h['close'].values > ema_50_12h
    downtrend_12h = df_12h['close'].values < ema_50_12h
    uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h)
    downtrend_12h_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 0:
            # LONG: price > KAMA, volume confirmation, 12h uptrend
            if close[i] > kama[i] and vol_confirm and uptrend_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price < KAMA, volume confirmation, 12h downtrend
            elif close[i] < kama[i] and vol_confirm and downtrend_12h_aligned[i]:
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