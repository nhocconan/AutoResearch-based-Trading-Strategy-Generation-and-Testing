#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_Chop_Filter_v4
Hypothesis: KAMA identifies trend direction on daily timeframe, RSI(14) filters for momentum strength, and Choppiness Index(14) identifies ranging markets. Enter long when KAMA up, RSI>50, and CHOP<38.2 (trending). Enter short when KAMA down, RSI<50, and CHOP<38.2. Avoid trades when CHOP>61.8 (ranging). Designed for 1d timeframe to capture multi-day trends with low trade frequency, suitable for both bull and bear markets.
"""

name = "1d_KAMA_Direction_RSI_Chop_Filter_v4"
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
    
    # KAMA (Kaufman Adaptive Moving Average) - trend direction
    def kama(price, period=10, fast=2, slow=30):
        change = np.abs(np.diff(price, n=period))
        volatility = np.sum(np.abs(np.diff(price)), axis=1)
        er = np.zeros_like(price)
        er[period:] = change[period-1:] / (volatility[period-1:] + 1e-10)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(price)
        kama[period] = price[period]
        for i in range(period+1, len(price)):
            kama[i] = kama[i-1] + sc[i] * (price[i] - kama[i-1])
        return kama
    
    kama_vals = kama(close, period=10, fast=2, slow=30)
    
    # RSI(14) - momentum strength
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # Choppiness Index(14) - regime filter
    def chop(high, low, close, period=14):
        atr = []
        for i in range(1, len(high)):
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            atr.append(tr)
        atr = np.array(atr)
        sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
        hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
        ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
        chop = 100 * np.log10(sum_atr / (hh - ll)) / np.log10(period)
        return np.concatenate([np.full(period, np.nan), chop])
    
    chop_vals = chop(high, low, close, period=14)
    
    signals = np.zeros(n)
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any indicator is not ready
        if np.isnan(kama_vals[i]) or np.isnan(rsi[i]) or np.isnan(chop_vals[i]):
            signals[i] = 0.0
            continue
            
        # Long conditions: KAMA up (price > KAMA), RSI > 50, trending market (CHOP < 38.2)
        if close[i] > kama_vals[i] and rsi[i] > 50 and chop_vals[i] < 38.2:
            signals[i] = 0.25
        # Short conditions: KAMA down (price < KAMA), RSI < 50, trending market (CHOP < 38.2)
        elif close[i] < kama_vals[i] and rsi[i] < 50 and chop_vals[i] < 38.2:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals