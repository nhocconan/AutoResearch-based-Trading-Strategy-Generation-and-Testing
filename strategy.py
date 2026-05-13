#!/usr/bin/env python3
# 4h_SuperTrend_EMA_Crossover_With_Volume_Confirmation
# Hypothesis: In both bull and bear markets, price tends to respect the SuperTrend direction
# when aligned with EMA crossover signals. Volume confirmation filters out false signals.
# SuperTrend adapts to volatility via ATR, making it effective in ranging and trending markets.
# EMA crossover (9/21) provides timely entries while SuperTrend (10,3) filters for trend strength.
# This combination aims to capture medium-term moves with limited whipsaw, keeping trade frequency low.

name = "4h_SuperTrend_EMA_Crossover_With_Volume_Confirmation"
timeframe = "4h"
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

    # Calculate ATR for SuperTrend
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values

    # SuperTrend calculation
    hl2 = (high + low) / 2
    upperband = hl2 + (3.0 * atr)
    lowerband = hl2 - (3.0 * atr)
    
    supertrend = np.zeros_like(close)
    dir = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = hl2[0]
    dir[0] = 1
    
    for i in range(1, n):
        if close[i] > upperband[i-1]:
            dir[i] = 1
        elif close[i] < lowerband[i-1]:
            dir[i] = -1
        else:
            dir[i] = dir[i-1]
            if dir[i] == 1 and lowerband[i] < lowerband[i-1]:
                lowerband[i] = lowerband[i-1]
            if dir[i] == -1 and upperband[i] > upperband[i-1]:
                upperband[i] = upperband[i-1]
        
        if dir[i] == 1:
            supertrend[i] = lowerband[i]
        else:
            supertrend[i] = upperband[i]

    # EMA crossover (9/21) for entry timing
    ema9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values

    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(21, n):  # Start after EMA21 warmup
        if position == 0:
            # LONG: Uptrend (SuperTrend) + bullish EMA crossover + volume spike
            if dir[i] == 1 and ema9[i] > ema21[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend (SuperTrend) + bearish EMA crossover + volume spike
            elif dir[i] == -1 and ema9[i] < ema21[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend turns bearish or EMA crossover reverses
            if dir[i] == -1 or ema9[i] < ema21[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend turns bullish or EMA crossover reverses
            if dir[i] == 1 or ema9[i] > ema21[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals