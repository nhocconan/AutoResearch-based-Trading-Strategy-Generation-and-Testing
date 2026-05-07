#!/usr/bin/env python3
"""
4h_KAMA_Direction_With_RSI_and_Chop_Filter
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) to determine trend direction,
combine with RSI for momentum strength and Chop Index for regime filtering.
Only take long when KAMA is rising, RSI > 50, and Chop > 61.8 (ranging).
Only take short when KAMA is falling, RSI < 50, and Chop > 61.8 (ranging).
This avoids trending markets where whipsaws occur, focusing on mean reversion in ranging markets.
Designed for 20-30 trades/year on 4h timeframe to minimize fee drag.
"""

name = "4h_KAMA_Direction_With_RSI_and_Chop_Filter"
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
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0]))[:, None], axis=1)
    # Fix: calculate volatility as rolling sum of absolute changes
    volatility = pd.Series(change).rolling(window=10, min_periods=1).sum().values
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    sc = np.power(er * (2/(2+1) - 2/(30+1)) + 2/(30+1), 2)  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Chop Index (14)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]  # first period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Max and min close over 14 periods
    max_h = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_l = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * np.log10((atr * 14) / (max_h - min_l)) / np.log10(14)
    # Handle division by zero or invalid cases
    chop = np.where((max_h - min_l) > 0, chop, 50.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for KAMA, RSI, Chop
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine KAMA direction (rising/falling)
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        if position == 0:
            # Long: KAMA rising, RSI > 50, Chop > 61.8 (ranging market)
            if kama_rising and rsi[i] > 50 and chop[i] > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling, RSI < 50, Chop > 61.8 (ranging market)
            elif kama_falling and rsi[i] < 50 and chop[i] > 61.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA turns falling or Chop drops below 61.8 (trending)
            if not kama_rising or chop[i] < 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA turns rising or Chop drops below 61.8 (trending)
            if not kama_falling or chop[i] < 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals