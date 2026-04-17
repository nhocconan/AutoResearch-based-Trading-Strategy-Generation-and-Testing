#!/usr/bin/env python3
"""
1d_KAMA_RSI_ChopFilter_v1
1d strategy: Use KAMA direction for trend, RSI(14) for mean reversion entries,
and Choppiness Index (14) for regime filter. Enter long when KAMA up, RSI<30, CHOP>61.8.
Enter short when KAMA down, RSI>70, CHOP>61.8. Exit when RSI crosses 50.
Designed to capture mean reversion in ranging markets while avoiding strong trends.
Target: 30-100 total trades over 4 years (7-25/year).
"""

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
    
    # === KAMA (14, 2, 30) for trend direction ===
    # Efficiency Ratio
    change = np.abs(close - np.roll(close, 14))
    change[0] = 0
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None
    # Manual calculation for volatility sum over 14 periods
    volatility = np.zeros_like(close)
    for i in range(14, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-13:i+1])))
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI(14) for mean reversion signal ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Choppiness Index (14) for regime filter ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Chop formula
    chop = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(14)
    # Handle division by zero when hh == ll
    chop = np.where((hh - ll) != 0, chop, 50)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # KAMA direction: compare to previous value
        kama_up = kama[i] > kama[i-1]
        kama_down = kama[i] < kama[i-1]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: KAMA up, RSI < 30 (oversold), Chop > 61.8 (ranging market)
            if (kama_up and 
                rsi[i] < 30 and 
                chop[i] > 61.8):
                signals[i] = 0.25
                position = 1
                continue
            # Short: KAMA down, RSI > 70 (overbought), Chop > 61.8 (ranging market)
            elif (kama_down and 
                  rsi[i] > 70 and 
                  chop[i] > 61.8):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: RSI crosses 50 (mean reversion complete)
        elif position == 1:
            # Exit long: RSI crosses above 50
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI crosses below 50
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0