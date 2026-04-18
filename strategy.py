#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_ChopFilter
4h strategy using KAMA direction, RSI momentum filter, and Choppiness index regime filter.
- Long: KAMA rising + RSI > 50 + Choppiness > 61.8 (ranging market)
- Short: KAMA falling + RSI < 50 + Choppiness > 61.8 (ranging market)
- Exit: Opposite signal
Designed for mean reversion in ranging markets with trend momentum confirmation.
Works in both bull and bear markets by focusing on ranging conditions.
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
    
    # KAMA calculation
    def calculate_kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        # Avoid division by zero
        er = np.where(volatility != 0, change / volatility, 0)
        # Smoothing constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA
        kama = np.full_like(close, np.nan, dtype=float)
        kama[length-1] = close[length-1]
        for i in range(length, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Choppiness Index
    def calculate_chop(high, low, close, length=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period TR is just high-low
        
        # Sum of True Range over period
        atr_sum = np.zeros_like(close)
        for i in range(length-1, len(close)):
            atr_sum[i] = np.sum(tr[i-length+1:i+1])
        
        # Highest high and lowest low over period
        highest_high = np.zeros_like(close)
        lowest_low = np.zeros_like(close)
        for i in range(length-1, len(close)):
            highest_high[i] = np.max(high[i-length+1:i+1])
            lowest_low[i] = np.min(low[i-length+1:i+1])
        
        # Choppiness Index
        chop = np.full_like(close, 50.0, dtype=float)
        for i in range(length-1, len(close)):
            if highest_high[i] != lowest_low[i]:
                chop[i] = 100 * np.log10(atr_sum[i] / (highest_high[i] - lowest_low[i])) / np.log10(length)
        return chop
    
    # Calculate indicators
    kama = calculate_kama(close, length=10, fast=2, slow=30)
    rsi = np.full_like(close, 50.0, dtype=float)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = np.full_like(close, 0.0, dtype=float)
    avg_loss = np.full_like(close, 0.0, dtype=float)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    chop = calculate_chop(high, low, close, length=14)
    
    # KAMA direction
    kama_rising = kama > np.roll(kama, 1)
    kama_falling = kama < np.roll(kama, 1)
    kama_rising[0] = False
    kama_falling[0] = False
    
    # RSI conditions
    rsi_above_50 = rsi > 50
    rsi_below_50 = rsi < 50
    
    # Chop condition (ranging market)
    chop_high = chop > 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need enough for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA rising + RSI > 50 + Chop > 61.8 (ranging market)
            if kama_rising[i] and rsi_above_50[i] and chop_high[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling + RSI < 50 + Chop > 61.8 (ranging market)
            elif kama_falling[i] and rsi_below_50[i] and chop_high[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Opposite signal
            if kama_falling[i] and rsi_below_50[i] and chop_high[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Opposite signal
            if kama_rising[i] and rsi_above_50[i] and chop_high[i]:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Direction_RSI_ChopFilter"
timeframe = "4h"
leverage = 1.0