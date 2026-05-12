#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_ChopFilter
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise—trending in low volatility, flattening in high volatility. 
Combined with RSI for overbought/oversold conditions and Choppiness Index to filter ranging markets, this strategy aims to capture 
strong trending moves while avoiding false signals in chop. Works in bull/bear by following KAMA direction only when market is trending.
"""

name = "4h_KAMA_Direction_RSI_ChopFilter"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === KAMA (10, 2, 30) on close ===
    # Efficiency Ratio (ER) = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(close - np.roll(close, 10))
    vol = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # temporary fix, will replace with loop below
    # Recompute volatility correctly using a loop for rolling sum of absolute changes
    vol = np.zeros(n)
    for i in range(10, n):
        vol[i] = np.sum(np.abs(close[i-9:i+1] - np.roll(close[i-9:i+1], 1)))
    er = np.zeros(n)
    er[10:] = change[10:] / np.where(vol[10:] == 0, 1, vol[10:])  # avoid div by zero
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI(14) ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    for i in range(15, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # === Get 1d data for Choppiness Index ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range and ATR(14) for Chop
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.zeros_like(tr)
    for i in range(14, len(tr)):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Sum of True Range over 14 periods
    sum_tr = np.zeros_like(tr)
    for i in range(14, len(tr)):
        sum_tr[i] = np.sum(tr[i-13:i+1])
    
    # Choppiness Index = 100 * log10(sum_tr / (atr * 14)) / log10(14)
    chop = np.zeros_like(tr)
    for i in range(14, len(tr)):
        if atr[i] > 0:
            chop[i] = 100 * np.log10(sum_tr[i] / (atr[i] * 14)) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    
    # Align KAMA, RSI, Chop to 4h
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop, additional_delay_bars=0)  # Chop is contemporaneous
    
    # === Signals ===
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # warmup for KAMA/RSI/Chop
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price > KAMA AND RSI > 50 AND Chop < 38.2 (trending)
            if (close[i] > kama_aligned[i] and 
                rsi_aligned[i] > 50 and 
                chop_aligned[i] < 38.2):
                signals[i] = 0.25
                position = 1
            # SHORT: Price < KAMA AND RSI < 50 AND Chop < 38.2 (trending)
            elif (close[i] < kama_aligned[i] and 
                  rsi_aligned[i] < 50 and 
                  chop_aligned[i] < 38.2):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price < KAMA OR Chop > 61.8 (chop)
            if (close[i] < kama_aligned[i]) or (chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price > KAMA OR Chop > 61.8 (chop)
            if (close[i] > kama_aligned[i]) or (chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals