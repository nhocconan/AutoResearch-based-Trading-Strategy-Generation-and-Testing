#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI + Chop Regime
# Uses Kaufman's Adaptive Moving Average for trend direction, RSI for momentum, and Choppiness Index for regime filtering.
# Long when KAMA trending up, RSI > 50, and Chop > 61.8 (ranging market for mean reversion).
# Short when KAMA trending down, RSI < 50, and Chop > 61.8.
# Exit when Chop < 38.2 (trending market) or RSI crosses 50.
# Designed for 1d timeframe to avoid overtrading, with Chop filter to avoid false signals in strong trends.
# Target: 30-100 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (adaptive moving average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.zeros_like(change)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start at index 9
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i-10] * (close[i] - kama[i-1])
    
    # Calculate RSI (14-period)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # Calculate Choppiness Index (14-period)
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
    # Chop = 100 * log10(tr_sum / (hh - ll)) / log10(14)
    chop = 100 * np.log10(tr_sum / (hh - ll + 1e-10)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    for i in range(14, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            continue
        
        # Long entry: KAMA up, RSI > 50, Chop > 61.8 (ranging market)
        if (close[i] > kama[i] and rsi[i] > 50 and chop[i] > 61.8 and position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: KAMA down, RSI < 50, Chop > 61.8 (ranging market)
        elif (close[i] < kama[i] and rsi[i] < 50 and chop[i] > 61.8 and position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Chop < 38.2 (trending market) or RSI crosses 50
        elif position == 1 and (chop[i] < 38.2 or rsi[i] < 50):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (chop[i] < 38.2 or rsi[i] > 50):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_KAMA_RSI_Chop_Regime"
timeframe = "1d"
leverage = 1.0