#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_RSI_Filter_and_Volume_Confirmation
Hypothesis: Use KAMA to determine trend direction, RSI for overbought/oversold conditions, and volume surge for confirmation. 
Go long when KAMA is rising, RSI > 50, and volume > 1.5x average. Go short when KAMA is falling, RSI < 50, and volume > 1.5x average.
Designed to capture momentum in both bull and bear markets with controlled risk via tight entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER = Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if len(change) > 1 else np.abs(change[0])
    # Simplified ER calculation for series
    er = np.zeros(n)
    for i in range(10, n):  # ER window of 10
        if i >= 10:
            price_change = np.abs(close[i] - close[i-10])
            price_volatility = np.sum(np.abs(np.diff(close[i-9:i+1])))
            if price_volatility > 0:
                er[i] = price_change / price_volatility
            else:
                er[i] = 1.0
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # Fast=2, Slow=30
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate average volume (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # need volume MA and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i-1]) if i > 0 else True) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: KAMA rising, RSI > 50, volume confirmation
            if kama[i] > kama[i-1] and rsi[i] > 50 and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: KAMA falling, RSI < 50, volume confirmation
            elif kama[i] < kama[i-1] and rsi[i] < 50 and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: KAMA falling or RSI < 50
            if kama[i] < kama[i-1] or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA rising or RSI > 50
            if kama[i] > kama[i-1] or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Trend_With_RSI_Filter_and_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0