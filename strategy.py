#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_RSI_Filter_and_Volume_Confirmation
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) to capture trend direction, filtered by RSI for overbought/oversold conditions and volume confirmation for momentum. Go long when price crosses above KAMA with RSI < 70 and volume > 1.5x average, short when price crosses below KAMA with RSI > 30 and volume > 1.5x average. This strategy aims to capture trend moves while avoiding extremes and ensuring momentum confirmation, suitable for both bull and bear markets. Designed for 4h timeframe to limit trades and avoid fee drag.
"""

name = "4h_KAMA_Trend_With_RSI_Filter_and_Volume_Confirmation"
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
    # Efficiency ratio
    change = np.abs(np.diff(close, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # start at index 9 for 10-period lookback
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI (14-period)
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
    
    # Calculate volume average (20-period)
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(kama[i-1]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: price crosses above KAMA + RSI < 70 (not overbought) + volume confirmation
            if close[i-1] <= kama[i-1] and close[i] > kama[i] and rsi[i] < 70 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # SHORT: price crosses below KAMA + RSI > 30 (not oversold) + volume confirmation
            elif close[i-1] >= kama[i-1] and close[i] < kama[i] and rsi[i] > 30 and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below KAMA or RSI > 70 (overbought)
            if close[i] < kama[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above KAMA or RSI < 30 (oversold)
            if close[i] > kama[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals