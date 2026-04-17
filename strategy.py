#!/usr/bin/env python3
"""
4h_KAMA_Trend_RSI_Momentum
Strategy: KAMA trend direction with RSI momentum filter and volume confirmation.
Long: KAMA rising + RSI > 55 + volume > 1.5x MA20
Short: KAMA falling + RSI < 45 + volume > 1.5x MA20
Exit: Opposite RSI signal or volume drop
Position size: 0.25
Designed to capture momentum in trending markets while avoiding chop.
Timeframe: 4h
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
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    close_series = pd.Series(close)
    # Efficiency Ratio
    change = abs(close_series - close_series.shift(10))
    volatility = abs(close_series.diff()).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    # Smoothing constants
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc.iloc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation (20-period MA)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (i < 10 or np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction: rising if current > previous
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Entry conditions
        if position == 0:
            # Long: KAMA rising + RSI > 55 + volume
            if (kama_rising and rsi[i] > 55 and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling + RSI < 45 + volume
            elif (kama_falling and rsi[i] < 45 and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI drops below 50 or volume filter fails
            if rsi[i] < 50 or not volume_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI rises above 50 or volume filter fails
            if rsi[i] > 50 or not volume_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Trend_RSI_Momentum"
timeframe = "4h"
leverage = 1.0