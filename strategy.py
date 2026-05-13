#!/usr/bin/env python3
"""
12h_KAMA_Direction_RSI_ChopFilter
Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both bull and bear markets. 
When combined with RSI momentum and a Choppiness regime filter (trend when CHOP < 38.2, range when > 61.8), 
it filters false signals. Entry occurs when KAMA direction aligns with RSI > 50 (bullish) or < 50 (bearish) 
in the correct regime, with volume confirmation. Designed for low trade frequency (~15-25/year) on 12h.
"""

name = "12h_KAMA_Direction_RSI_ChopFilter"
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
    
    # === KAMA (adaptive trend) on 12h close ===
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum |close[t] - close[t-1]| over 10
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # start after 10 periods
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI (14) on 12h close ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Choppiness Index (14) on 1d HTF ===
    df_1d = get_htf_data(prices, '1d')
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # ATR(14)
    atr = np.full_like(tr, np.nan)
    atr[13] = np.mean(tr[1:14])
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    # Sum of ATR over 14 periods
    sum_atr = np.full_like(tr, np.nan)
    for i in range(13, len(tr)):
        sum_atr[i] = np.sum(tr[i-13:i+1])
    # Choppiness
    chop = np.full_like(tr, np.nan)
    for i in range(13, len(tr)):
        if sum_atr[i] > 0:
            chop[i] = 100 * np.log10(atr[i] / sum_atr[i]) / np.log10(14)
        else:
            chop[i] = 50
    # Align chop to 12h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === Volume confirmation: current > 1.5x 20-period average ===
    vol_ma = np.full_like(volume, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = max(20, 14, 13)  # KAMA(10) needs 10, RSI(14) needs 14, CHOP(14) needs 13
    
    for i in range(start_idx, n):
        if position == 0:
            # Determine regime: trending (CHOP < 38.2) or ranging (CHOP > 61.8)
            # Only trade in trending regime for trend-following logic
            if chop_aligned[i] < 38.2:  # Trending regime
                # LONG: KAMA up (close > KAMA), RSI > 50, volume confirmation
                if (close[i] > kama[i] and 
                    rsi[i] > 50 and 
                    volume_filter[i]):
                    signals[i] = 0.25
                    position = 1
                # SHORT: KAMA down (close < KAMA), RSI < 50, volume confirmation
                elif (close[i] < kama[i] and 
                      rsi[i] < 50 and 
                      volume_filter[i]):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # EXIT LONG: KAMA reverses down OR chop exits trending regime
            if (close[i] < kama[i] or 
                chop_aligned[i] >= 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA reverses up OR chop exits trending regime
            if (close[i] > kama[i] or 
                chop_aligned[i] >= 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals