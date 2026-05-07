#!/usr/bin/env python3
# 1d_KAMA_RSI_Chop_Filter
# Hypothesis: Uses daily KAMA for trend direction, RSI for overbought/oversold levels, and Choppiness Index to filter ranging markets.
# Long when: price > KAMA, RSI < 30 (oversold), and CHOP > 61.8 (ranging market) for mean reversion.
# Short when: price < KAMA, RSI > 70 (overbought), and CHOP > 61.8 (ranging market) for mean reversion.
# Designed for low trade frequency (10-25/year) to minimize fee drag while capturing mean reversion in ranging markets.
# Works in both bull and bear markets by focusing on mean reversion during ranging periods.

name = "1d_KAMA_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA calculation (ER = 10, fast = 2, slow = 30)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI calculation (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index calculation (14-period)
    atr = np.zeros_like(close)
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.zeros_like(close)
    for i in range(14, len(close)):
        if max_high[i] != min_low[i]:
            chop[i] = 100 * np.log10(np.sum(atr[i-13:i+1]) / np.log10(14) / (max_high[i] - min_low[i]))
        else:
            chop[i] = 50  # neutral when no range
    
    signals = np.zeros(n)
    
    for i in range(14, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop[i])):
            continue
        
        # Long: price > KAMA, RSI < 30 (oversold), CHOP > 61.8 (ranging market)
        if close[i] > kama[i] and rsi[i] < 30 and chop[i] > 61.8:
            signals[i] = 0.25
        # Short: price < KAMA, RSI > 70 (overbought), CHOP > 61.8 (ranging market)
        elif close[i] < kama[i] and rsi[i] > 70 and chop[i] > 61.8:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals